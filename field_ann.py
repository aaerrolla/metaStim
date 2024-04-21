import os
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler



class FieldANNModel:
    def __init__(self, ec):
        self.ec = ec        

    def load_model(self):
        dataDir = os.path.join(os.getcwd(), "field-ann-models")
        nElecOn = np.sum(np.abs(ec))  # Total number of electrodes on
        modelPath = os.path.join(dataDir, f'ann-field-ec{nElecOn}-settings.json')
        weightPath = os.path.join(dataDir, f'ann-field-ec{nElecOn}-weights.h5')
        stdscaPath = os.path.join(dataDir, f'ann-field-ec{nElecOn}-input-std.bin')
        
        with open(modelPath, 'r') as f:
            model_json = f.read()
            self.model = model_from_json(model_json)
        
        self.model.load_weights(weightPath)
        self.std_scaler = load(stdscaPath)

    def predict_field(self, z, x=1, y=1):         
        xyz = np.column_stack((x*np.ones(z.shape), y*np.ones(z.shape), z))
        nPts = z.shape[0]
        xModel_raw = np.column_stack((np.tile(self.ec, (nPts, 1)), xyz))
        xModel = self.std_scaler.transform(xModel_raw)
        yModel = np.exp(self.model.predict(xModel).reshape(-1)) - 1
        return yModel
    
    def visualize_field(self, z, stimAmp):
        font = {'family': 'serif', 'color': 'black', 'size': 20}
        plt.plot(z, stimAmp * self.predict_field(z), 'k-', linewidth=1)
        plt.title('Sample field calculation', fontdict=font)
        plt.xlabel('z (mm)', fontdict=font)
        plt.ylabel('$\Phi$ (V)', fontdict=font)
        plt.show()


if __name__ == "__main__":
    ec = np.array([0, 1, 1, 1, 1, 1, 1, 0])  # Electrode configuration (+1, -1, or 0)
    stimAmp = 3  # Stimulation amplitude in Volts

    # Specify z values for field calculation
    z = np.linspace(-5, 16, num=100)

    print(f"z.shape : {z.shape}")
    x = 1 * np.ones(z.shape);
    print(f"x.shape : {x.shape}")
    
    y = 1 * np.ones(z.shape);
    print(f"y.shape : {y.shape}")

    xyz = np.column_stack((x, y, z))
    print(f"xyz.shape : {xyz.shape}")

    # Create an instance of FieldCalculator class
    field_calculator = FieldANNModel(ec)

    # Load the model
    field_calculator.load_model()

    # Visualize the field calculation
    field_calculator.visualize_field(z, stimAmp)
