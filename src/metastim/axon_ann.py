import numpy as np
import sys
from importlib import resources
from keras.models import model_from_json
from joblib import dump, load

from metastim import field_ann 
from metastim.utils import MetaStimUtil
from metastim import visualization as vis



class AxonANN:
    """calculates the voltage required to activate the axons of neurons"""

    def __init__(self, electrode_list, pulse_width , stimulation_amp, num_axons=10, axon_diameter=6):
        self.electrode_list = electrode_list
        self._validate_electrode_list(electrode_list)        
        self.pulse_width = pulse_width
        self._validate_pulse_width(pulse_width)        
        self.stimulation_amp = stimulation_amp
        self._validate_stimulation_amp(stimulation_amp)        
        # TODO: validation for num_axons
        self.num_axons = num_axons
        self._validate_num_axons(num_axons)
        self.axon_diameter = axon_diameter
        self._validate_axon_diameter(axon_diameter)
    
    def axon_ann(self, x_axon, y_axon, z_axon, lead_radius, threshold = None):
        """Predict axon activation based on electric potentials
           Output: (axon_activation, axon_thresholds) 
        """                
        field_ann_model = field_ann.FieldANN(self.electrode_list)
        phi_axon = field_ann_model.field_ann(x_axon, y_axon, z_axon)
        
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
        sd_11_axon = MetaStimUtil.get_field_sd(self.num_axons, phi_axon)

        # fx_axon
        fs_axon = MetaStimUtil.get_field_shape(self.num_axons, sd_11_axon)

        # axon_distance
        axon_distance = MetaStimUtil.get_axon_to_lead_dist(lead_radius, x_axon, y_axon)

        self._validate_axon_distance(axon_distance)
        
        # organize inputs to Axon ANN
        o = np.ones((self.num_axons,))
        x_axon_ann_raw = np.column_stack((fs_axon, o * self.axon_diameter, o * self.pulse_width, axon_distance, np.transpose(sd_11_axon)))

        # standardize inputs for Axon ANN
        x_axon_ann = sc_axon.transform(x_axon_ann_raw)

        # evaluate the Axon ANN model
        y_axon_ann = np.exp(axon_model.predict(x_axon_ann).reshape(-1))        
        axon_activation = (y_axon_ann <= self.stimulation_amp).astype(int)        

        if threshold == None:
            threshold = False

        if threshold:
            axon_thresholds = (y_axon_ann <= self.stimulation_amp).astype(float)
            return axon_activation, axon_thresholds
        else:
            return axon_activation, 
        

    def __repr__(self):
        properties = ", ".join(f"{key}='{value}'" for key, value in vars(self).items())
        return f"{type(self).__name__}({properties})"
    
    def __str__(self):        
        return self.__repr__()
    
        
    def _validate_axon_diameter(self, axon_diameter):
        """Axon Diameter(D) validate:
            if axon_diameter <= 0  print error and exit 
            if  1.5 > axon_diameter < 15  print Warning and continue
        """
        if axon_diameter <= 0:
            print('Error: Negative or Zero fiber diameter (D), D must be non-zero and positive (> 0).')
            sys.exit(1)        
        if axon_diameter < 1.5 or axon_diameter > 15:
            print('Warning! Accuracy may be degraded for fiber diameters outside of 1.5-15um.')              

    def _validate_pulse_width(self, pulse_width):        
        """stimulus pulse width(pw)  pulse_width validation:
            if pw <= 0  print Error and exit
            if  30 < pw > 500 : issue Warning continue
        """ 
        if pulse_width <= 0:
            print('Error: Negative or Zero pulse width (PW)! PW must be non-zero and positive (> 0).')
            sys.exit(2)        
        if pulse_width < 30 or pulse_width > 500:
            print('Warning! Accuracy may be degraded for pulse widths outside of 30-500us.')            
        
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
        
    def _validate_stimulation_amp(self, stimulation_amp):
        """use magnitude of stimulation_amp
        """
        self.stimulation_amp = abs(stimulation_amp)

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

    lead_radius = MetaStimUtil.get_lead_radius(lead_id, electrode_list)

    inl = 100 * axon_diameter / 1e3 # distance between nodes on an axon

    z_base = np.arange(-5, 16, inl)
    num_axon_nodes = z_base.shape[0]

    x_axon = np.repeat(np.linspace(min_distance, max_distance, num=num_axons), num_axon_nodes).reshape(num_axon_nodes, num_axons, order='F') + lead_radius
    y_axon = np.zeros(x_axon.shape)
    z_axon = np.repeat(z_base, num_axons).reshape(num_axon_nodes, num_axons)

    field_ann_model = field_ann.FieldANN(electrode_list)    
    axon_ann_model = AxonANN(electrode_list, pulse_width, stimulation_amp, num_axons, axon_diameter)
    
    phi_axon = field_ann_model.field_ann(x_axon, y_axon, z_axon)
    axon_act = axon_ann_model.axon_ann(x_axon, y_axon, z_axon, lead_radius)

    visualization = vis.Visualization(lead_id, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_act)
    visualization.visualize1(electrode_list)

if __name__ == "__main__":
    main()