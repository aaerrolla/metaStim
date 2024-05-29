from importlib import resources
from joblib import load
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from metastim import validations
import numpy as np
import matplotlib.pyplot as plt
import sys


class FieldANN:
    def __init__(self, electrode_config):
        if len(electrode_config) == 0:
            print("Invalid electrode configuration. Elements configuration is empty.")
            sys.exit(4)
        else:                
            validations.validate_electrode_list(electrode_config)
            self._electrode_config = electrode_config
            self._model = None
            self._std_scaler = None
            self.__load_model()

        
    def __load_model(self):
        """loads field ANN files and creates model"""
        num_elec_on = np.sum(np.abs(self._electrode_config))  # Total number of electrodes on
        model_file =  f'ann-field-ec{num_elec_on}-settings.json'  
        weight_file = f'ann-field-ec{num_elec_on}-weights.h5'
        std_sca_file = f'ann-field-ec{num_elec_on}-input-std.bin'
        
        with resources.open_text("metastim.field-ann-models", model_file) as f:
            model_json = f.read()
            self._model = model_from_json(model_json)
        
        with resources.open_binary("metastim.field-ann-models", weight_file) as wf:
            self._model.load_weights(wf.name)              

        with resources.open_binary("metastim.field-ann-models", std_sca_file) as ssf:            
            self._std_scaler = load(ssf.name)

    def predict_field(self, x, y, z):
        """evluate the model 
           Args:
            x: x vector  
            y: y vector
            z: z vector 
           Returns:
            y_model 
        """         
        xyz = np.column_stack((x, y, z))
        num_points = z.shape[0]
        x_model_raw = np.column_stack((np.tile(self._electrode_config, (num_points, 1)), xyz))
        x_model = self._std_scaler.transform(x_model_raw)
        y_model = np.exp(self._model.predict(x_model).reshape(-1)) - 1
        return y_model

    @property
    def electrode_config(self):
        return self._electrode_config
    
    @electrode_config.setter
    def electrode_config(self, value):
        self._electrode_config = value
        self.__load_model()
    
    def visualize_field(self, x, y, z, stim_amp):
        """visualize field using matplotlib
           Args:              
            x: x vector  
            y: y vector
            z: z vector
            stim_amp: stimulation amplitude            
           Returns:
            None
        """
        font = {'family': 'serif', 'color': 'black', 'size': 20}
        plt.plot(z, stim_amp * self.predict_field(x, y, z), 'k-', linewidth=1)
        plt.title('Sample field calculation', fontdict=font)
        plt.xlabel('z (mm)', fontdict=font)
        plt.ylabel('$\Phi$ (V)', fontdict=font)
        plt.show()


    def field_ann(self, x_axon, y_axon, z_axon, num_axons=10):
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
        ec = np.array(self._electrode_config) # electrode configuration (+1, -1, or 0)
        num_electrodes = ec.shape[0] # total number of electrodes
        num_electrodes_on = np.sum(np.abs(ec))
        # x_axon, y_axon, z_axon = self.axon_coord()
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

        for k in range(0, num_axons):
            # organize inputs
            num_nodes = x_axon[:,k].shape[0]
            xyz_axon = np.column_stack((x_axon[:,k], y_axon[:,k], z_axon[:,k]))
            x_field_raw = np.column_stack((np.tile(ec, (num_nodes,1)), xyz_axon)) 

            # standardize inputs
            x_field = sc_field.transform(x_field_raw)

            # evaluate the model
            y_field = np.exp(field_model.predict(x_field).reshape(-1)) - 1 
            phi_axon[:,k] = y_field

        return phi_axon
        


def main():
    electrode_config = np.array([0, 1, 1, 1, 1, 1, 1, 0])  # Electrode configuration (+1, -1, or 0)
    stim_amp = 3  # Stimulation amplitude in Volts

    # Specify x, y and z values for field calculation
    z = np.linspace(-5, 16, num=100)
    x = 1 * np.ones(z.shape)
    y = 1 * np.ones(z.shape)



    # Create an instance of FieldANN class
    field_calculator = FieldANN(electrode_config)

    # y_model = field_calculator.predict_field(x, y, z)
    
    # Visualize the field calculation

    field_calculator.visualize_field(x, y, z, stim_amp)

    electrode_config = np.array([1, 0, 0, 0, -1, 0, 0, 0])
    field_calculator.electrode_config = electrode_config
    field_calculator.visualize_field(x, y, z, stim_amp)

    field_calculator.visualize_field(x, y, z, 5)



if __name__ == "__main__":
    main()