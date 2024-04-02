from axon_ann import AxonANNModel
from visualization import Visualization
import os

if __name__ == "__main__":
    # lead_radius = 0.635 # [mm]
    lead_id  = '6172'
    electrode_list = [1, 0, 0, 0, 0, -1, 0, 0]
    stimulation_amp = 3 # [V]
    pulse_width = 90 #[us]
    num_axons = 10
    min_distance = 1
    max_distance = 5
    axon_diameter = 6 # [um]

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

    axon_ann_model = AxonANNModel(electrode_list, lead_id, num_axons, min_distance, max_distance, axon_diameter, pulse_width, stimulation_amp)

    x_axon, y_axon, z_axon = axon_ann_model.axon_coord()

    phi_axon = axon_ann_model.field_ann()
    axon_act = axon_ann_model.axon_ann()

    visualization = Visualization(lead_id, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_act)
    visualization.visualize()


    # axon_ann_model.electrode_list = [0, 1, 1, 1, 1, 1, 1, 0]
    # axon_ann_model.stimulation_amp = 10
    
    # x_axon, y_axon, z_axon = axon_ann_model.axon_coord()
    # phi_axon = axon_ann_model.field_ann()
    # axon_act = axon_ann_model.axon_ann()

    # visualization = Visualization(lead_radius, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_act)
    # visualization.visualize()