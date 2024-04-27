import os
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

import validations

class FieldANNModel:
    def __init__(self, electrode_config):
        self.electrode_config = electrode_config
        validations.validate_electrode_list(electrode_config)

    def load_model(self):
        """loads field ANN files and creates model"""
        data_dir = os.path.join(os.getcwd(), "field-ann-models")
        num_elec_on = np.sum(np.abs(self.electrode_config))  # Total number of electrodes on
        model_path = os.path.join(data_dir, f'ann-field-ec{num_elec_on}-settings.json')
        weight_path = os.path.join(data_dir, f'ann-field-ec{num_elec_on}-weights.h5')
        std_sca_path = os.path.join(data_dir, f'ann-field-ec{num_elec_on}-input-std.bin')
        
        with open(model_path, 'r') as f:
            model_json = f.read()
            self.model = model_from_json(model_json)
        
        self.model.load_weights(weight_path)
        self.std_scaler = load(std_sca_path)

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
        x_model_raw = np.column_stack((np.tile(self.electrode_config, (num_points, 1)), xyz))
        x_model = self.std_scaler.transform(x_model_raw)
        y_model = np.exp(self.model.predict(x_model).reshape(-1)) - 1
        return y_model
    
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


if __name__ == "__main__":
    electrode_config = np.array([0, 1, 1, 1, 1, 1, 1, 0])  # Electrode configuration (+1, -1, or 0)
    stim_amp = 3  # Stimulation amplitude in Volts

    # Specify x, y and z values for field calculation
    z = np.linspace(-5, 16, num=100)
    x = 1 * np.ones(z.shape);
    y = 1 * np.ones(z.shape);



    # Create an instance of FieldANNModel class
    field_calculator = FieldANNModel(electrode_config)

    # Load the model
    field_calculator.load_model()

    # Visualize the field calculation
    field_calculator.visualize_field(x, y, z, stim_amp)
