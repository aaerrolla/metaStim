import numpy as np
import sys
from importlib import resources
from keras.models import model_from_json
from joblib import dump, load

from metastim import utils
from metastim import lead_selector
from metastim import visualization as vis



class AxonANNModel:
    """calculates the voltage required to activate the axons of neurons"""

    def __init__(self, lead_id, electrode_list, pulse_width , stimulation_amp, num_axons=10, min_distance=1, max_distance=5, axon_diameter=6):        
        self.axon_diameter = axon_diameter
        self._validate_axon_diameter(axon_diameter)
        self.pulse_width = pulse_width
        self._validate_pulse_width(pulse_width)        
        self.electrode_list = electrode_list
        self._validate_electrode_list(electrode_list)
        leadselector =  lead_selector.LeadSelector('DBSLead-smry.csv')
        self.leads = leadselector.load_leads();        
        self._validate_lead(lead_id)        
        # TODO: validation for num_axons
        self.num_axons = num_axons
        self._validate_num_axons(num_axons)
        self.min_distance = min_distance
        self._validate_min_distance(min_distance)
        self.max_distance = max_distance
        self._validate_max_distance(max_distance)        
        self.stimulation_amp = stimulation_amp
        self._validate_stimulation_amp(stimulation_amp)        
                
    
    
    def axon_coord(self):
        """This function gives user some axon coordinates to sample    
            num_axons, number of axon
            min_distance, minimum distance from the lead (default = 1mm)
            max_distance, maximum distance from the lead (default = 5mm)
            D, axon diameter (default = 6um)
            OUTPUT:
                x, y, and z coordinates of axon [in mm]
            NOTES:
                - all axons are parallel to the cylindrical lead; this is a simple example for now
                - future versions of code will pull axons from a data atlas depending on the brain location or application
                - it will be more efficient to return points in one data structure
        """
        inl = 100 * self.axon_diameter / 1e3 # distance between nodes on an axon

        z_base = np.arange(-5, 16, inl)
        num_axon_nodes = z_base.shape[0]

        x_axon = np.repeat(np.linspace(self.min_distance, self.max_distance, num=self.num_axons), num_axon_nodes).reshape(num_axon_nodes, self.num_axons, order='F') + self.lead_radius
        y_axon = np.zeros(x_axon.shape)
        z_axon = np.repeat(z_base, self.num_axons).reshape(num_axon_nodes, self.num_axons)

        return x_axon,y_axon,z_axon

    
    def field_ann(self):
        """This is a main function that calculates the electric potentials across axons
           Args:
           electrode_list: electrode configuration(s) (1, # electrodes) 0 is off, 1 is on and positive, -1 is on and negative
           axon_coord: xyz coordinates of each axon (3 x # points per axon x # axons)
           amp, stimulation amplitude in Volts
           lead, lead model (optional, Model 6172 is the selectable option right now)
           OUTPUT:
             phi, electric potentials from Field ANN for each axon [in V]
           NOTES:
                each axon in this demo has the same number of nodes / points
                the code should be generalized so that each axon can have different # pts / axon
                this could be done with a struct or a n x 4 matrix [axon ID, x, y, z], where n = # of points across all axons 
        """
        electrode_config = np.array(self.electrode_list) # electrode configuration (+1, -1, or 0)
        num_electrodes = electrode_config.shape[0] # total number of electrodes
        num_electrodes_on = np.sum(np.abs(electrode_config))
        x_axon, y_axon, z_axon = self.axon_coord()
        # directories and filenames

        # ----- Load Field ANN files ---
        field_ann_setting_file = f'ann-field-ec{num_electrodes_on}-settings.json'
        field_ann_weight_file = f'ann-field-ec{num_electrodes_on}-weights.h5'
        field_ann_std_file = f'ann-field-ec{num_electrodes_on}-input-std.bin'

        # ----- LOAD MODEL -----
        # load ann model
        
        with resources.open_text("metastim.field-ann-models", field_ann_setting_file) as settings_file:
            json_data = settings_file.read()
            field_model = model_from_json(json_data)

        #load weights
        with resources.open_binary("metastim.field-ann-models", field_ann_weight_file) as weight_file:
            field_model.load_weights(weight_file.name)

        # load standard scalar for inputs
        with resources.open_binary("metastim.field-ann-models", field_ann_std_file) as std_file:            
            sc_field = load(std_file.name)

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


    
    
    def axon_ann(self):
        """Predict axon activation based on electric potentials
           Output: axon activation
        """        
        
        x_axon, y_axon, z_axon = self.axon_coord()
        phi_axon = self.field_ann()

        # ----- LOAD Axon ANN Model -----
        # load ann model        
        with resources.open_text("metastim.axon-ann-model", "ann-axon-settings.json") as settings_file:
            json_data = settings_file.read()
            axon_model = model_from_json(json_data)

        #load weights
        with resources.open_text("metastim.axon-ann-model", "ann-axon-weights.h5") as weights_file:
            axon_model.load_weights(weights_file.name)

        # load standard scalar for inputs
        with resources.open_text("metastim.axon-ann-model", "ann-axon-input-std.bin") as std_file:
            sc_axon = load(std_file.name)

        # sd_11_axon
        sd_11_axon = utils.MetaStimUtil.get_field_sd(self.num_axons, phi_axon)

        # fx_axon
        fs_axon = utils.MetaStimUtil.get_field_shape(self.num_axons, sd_11_axon)

        # axon_distance
        axon_distance = utils.MetaStimUtil.get_axon_to_lead_dist(self.lead_radius, x_axon, y_axon)

        self._validate_axon_distance(axon_distance)
        
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
    def _validate_axon_diameter(self, axon_diameter):
        if axon_diameter < 0:
            print('Negative fiber diameter (D)! D must be positive (> 0).')
            print('setting axon_diameter  to 6.')
            self.axon_diameter = 6 # reset to default value and continue        
        
        if axon_diameter < 1.5 or axon_diameter > 15:
            print('Warning! Accuracy may be degraded for fiber diameters outside of 1.5-15um.')  
            sys.exit(1)

        
    # CHECK INPUTS
    # for this demo, there are no errors
    # However, checks need to be in place to let the user know what values are acceptable or not
    # D, fiber diameter
    def _validate_pulse_width(self, pulse_width):        
        # pw, stimulus pulse width
        if pulse_width < 0:
            print('Negative pulse width (PW)! PW must be positive (> 0).')
            sys.exit(2)
        # halt the code
        if pulse_width < 30 or pulse_width > 500:
            print('Warning! Accuracy may be degraded for pulse widths outside of 30-500us.')
            sys.exit(3)
        

    def _validate_electrode_list(self, electrode_list):
        for elec in electrode_list:
            if elec not in [-1, 0, 1]:
                print("Invalid electrode configuration. Elements must be either -1, 0, or 1.")
                sys.exit(4)
    
    def _validate_lead(self, lead_id):
        if lead_id not in self.leads.keys():
            print(f"Invalid lead specified. Lead Id must be  of {self.leads.keys()}.")
            sys.exit(5)
        else:
            lead = self.leads.get(lead_id)
            if lead.no != len(self.electrode_list) :
                print(f"Invalid electrode configuration. {lead_id} contains {lead.no} electrods")
                sys.exit(6)

            # get radius 
            self.lead_radius = lead.re

    def _validate_num_axons(self, num_axons):
        pass

    def _validate_min_distance(self, min_distance):
        pass

    def _validate_max_distance(self, max_distance):
        pass
        
    def _validate_stimulation_amp(self, stimulation_amp):
        pass

    def _validate_axon_distance(self, axon_distance):
        if (axon_distance.any() < 0.5  or axon_distance.any() > 9):
            print("Warning! Accuracy may be degraded as the minimum distance between axon and lead is out of range (0.5mm - 9mm)")
    

def main():
    """This exists for testing this module"""
    lead_id = '6172'
    electrode_list = [1, 0, 0, 0, -1, 0, 0, 0]
    stimulation_amp = 3
    pulse_width = 90
    num_axons = 10
    min_distance = 1
    max_distance = 5
    axon_diameter = 6

    axon_ann_model = AxonANNModel(lead_id, electrode_list,  pulse_width, stimulation_amp, num_axons, min_distance, max_distance, axon_diameter)

    x_axon, y_axon, z_axon = axon_ann_model.axon_coord()

    phi_axon = axon_ann_model.field_ann()
    axon_act = axon_ann_model.axon_ann()

    visualization = vis.Visualization(lead_id, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_act)
    visualization.visualize1(electrode_list)

if __name__ == "__main__":
    main()