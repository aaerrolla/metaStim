import numpy as np
import os
from keras.models import model_from_json
from joblib import dump, load

from utils import MetaStimUtil

# AxonANNModel : calculates the voltage required to activate the axons of neurons
class AxonANNModel:

    def __init__(self, electrode_list, lead_radius, num_axons, min_distance, max_distance, axon_diameter, pulse_width , stimulation_amp):
        self.axon_diameter = axon_diameter
        self.pulse_width = pulse_width
        self.validate_input(axon_diameter, pulse_width)
        self.electrode_list = electrode_list
        self.lead_radius = lead_radius
        self.num_axons = num_axons
        self.min_distance = min_distance
        self.max_distance = max_distance        
        self.stimulation_amp = stimulation_amp
                
    
    # Short description: this function gives user some axon coordinates to sample    
    # num_axons, number of axon
    # min_distance, minimum distance from the lead (default = 1mm)
    # max_distance, maximum distance from the lead (default = 5mm)
    # D, axon diameter (default = 6um)
    # OUTPUT:
    # x, y, and z coordinates of axon [in mm]
    # NOTES:
    # - all axons are parallel to the cylindrical lead; this is a simple example for now
    # - future versions of code will pull axons from a data atlas depending on the brain location or application
    # - it will be more efficient to return points in one data structure
    def axon_coord(self):
        inl = 100 * self.axon_diameter / 1e3 # distance between nodes on an axon

        z_base = np.arange(-5, 16, inl)
        num_axon_nodes = z_base.shape[0]

        x_axon = np.repeat(np.linspace(self.min_distance, self.max_distance, num=self.num_axons), num_axon_nodes).reshape(num_axon_nodes, self.num_axons, order='F') + self.lead_radius
        y_axon = np.zeros(x_axon.shape)
        z_axon = np.repeat(z_base, self.num_axons).reshape(num_axon_nodes, self.num_axons)

        return x_axon,y_axon,z_axon

    
    # field_ann
    # Short description: this is a main function that calculates the electric potentials across axons
    # e_config, electrode configuration(s) (1, # electrodes)
    # - 0 is off, 1 is on and positive, -1 is on and negative
    # ax_coord, xyz coordinates of each axon (3 x # points per axon x # axons)
    # amp, stimulation amplitude in Volts
    # lead, lead model (optional, Model 6172 is the selectable option right now)
    # OUTPUT:
    # phi, electric potentials from Field ANN for each axon [in V]
    # NOTES:
    # - each axon in this demo has the same number of nodes / points
    # - the code should be generalized so that each axon can have different # pts / axon
    # - this could be done with a struct or a n x 4 matrix [axon ID, x, y, z], where n = # of points across all axons 
    def field_ann(self):
        electrode_config = np.array(self.electrode_list) # electrode configuration (+1, -1, or 0)
        num_electrodes = electrode_config.shape[0] # total number of electrodes
        num_electrodes_on = np.sum(np.abs(electrode_config))
        x_axon, y_axon, z_axon = self.axon_coord()
        # directories and filenames
        data_dir = os.getcwd()

        # ----- Load Field ANN files ---
        field_ann_setting_file = data_dir + '/field-ann-models/ann-field-ec' + str(num_electrodes_on) + '-settings.json'
        field_ann_weight_file = data_dir + '/field-ann-models/ann-field-ec'  + str(num_electrodes_on) + '-weights.h5'
        field_ann_std_in_file = data_dir + '/field-ann-models/ann-field-ec' + str(num_electrodes_on) + '-input-std.bin'

        # ----- LOAD MODEL -----
        # load ann model
        #print('Loading model settings...')
        with open(field_ann_setting_file, 'r') as f:
            json_data = f.read()
        field_model = model_from_json(json_data)

        #load weights
        #print('Loading model weights...')
        field_model.load_weights(field_ann_weight_file)

        # load standard scalar for inputs
        #print('Loading input standarization fit...')
        sc_field = load(field_ann_std_in_file)

        # Calculate Potentials from Field ANN
        phi_axon = np.zeros(x_axon.shape)

        for k in range(0, self.num_axons):
            # organize inputs
            num_nodes = x_axon[:,k].shape[0]
            xyz_axon = np.column_stack((x_axon[:,k], y_axon[:,k], z_axon[:,k]))
            x_field_raw = np.column_stack((np.tile(electrode_config, (num_nodes,1)), xyz_axon)) 

            # standardize inputs
            x_field = sc_field.transform(x_field_raw)

            # evaluate the model
            y_field = np.exp(field_model.predict(x_field).reshape(-1)) - 1 
            phi_axon[:,k] = y_field

        return phi_axon


    # axon_ann 
    # Short Description : Predict axon activation based on electric potentials
    # Output: axon activation
    def axon_ann(self):
        
        data_dir = os.getcwd()
        axon_ann_setting_file = data_dir + '/axon-ann-model/ann-axon-settings.json'
        axon_ann_weight_file = data_dir + '/axon-ann-model/ann-axon-weights.h5'
        axon_ann_std_in_file = data_dir + '/axon-ann-model/ann-axon-input-std.bin'

        x_axon, y_axon, z_axon = self.axon_coord()
        phi_axon = self.field_ann()

        # ----- LOAD Axon ANN Model -----
        # load ann model
        # print('Loading Axon ANN settings...')
        with open(axon_ann_setting_file, 'r') as f:
            json_data = f.read()
        axon_model = model_from_json(json_data)

        #load weights
        # print('Loading Axon ANN weights...')
        axon_model.load_weights(axon_ann_weight_file)

        # load standard scalar for inputs
        # print('Loading Axon ANN input standarization...')
        sc_axon = load(axon_ann_std_in_file)

        # sd_11_axon
        sd_11_axon = MetaStimUtil.get_field_sd(self.num_axons, phi_axon)

        # fx_axon
        fs_axon = MetaStimUtil.get_field_shape(self.num_axons, sd_11_axon)

        # axon_distance
        axon_distance = MetaStimUtil.get_axon_to_lead_dist(self.lead_radius, x_axon, y_axon)

        # organize inputs to Axon ANN
        o = np.ones((self.num_axons,))
        x_axon_ann_raw = np.column_stack((fs_axon, o * self.axon_diameter, o * self.pulse_width, axon_distance, np.transpose(sd_11_axon)))

        # standardize inputs for Axon ANN
        x_axon_ann = sc_axon.transform(x_axon_ann_raw)

        # evaluate the Axon ANN model
        y_axon_ann = np.exp(axon_model.predict(x_axon_ann).reshape(-1))
        axon_activation = (y_axon_ann <= self.stimulation_amp).astype(int)

        return axon_activation

    def __repr__(self):
        properties = ", ".join(f"{key}='{value}'" for key, value in vars(self).items())
        return f"{type(self).__name__}({properties})"
    
    def __str__(self):        
        return self.__repr__()
    
    # CHECK INPUTS
    # for this demo, there are no errors
    # However, checks need to be in place to let the user know what values are acceptable or not
    # D, fiber diameter
    def validate_input(self, axon_diameter, pulse_width):
        if axon_diameter < 0:
            print('Negative fiber diameter (D)! D must be positive (> 0).')
            print('setting axon_diameter  to 6.')
            self.axon_diameter = 6 # reset to default value and continue
        if axon_diameter < 1.5 or axon_diameter > 15:
            print('Warning! Accuracy may be degraded for fiber diameters outside of 1.5-15um.')  
            exit(-1)

        # pw, stimulus pulse width
        if pulse_width < 0:
            print('Negative pulse width (PW)! PW must be positive (> 0).')
            exit(-2)
        # halt the code
        if pulse_width < 30 or axon_diameter > 500:
            print('Warning! Accuracy may be degraded for pulse widths outside of 30-500us.')
            exit(-3)
        