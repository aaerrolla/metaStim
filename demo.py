from metastim import field_ann, axon_ann
from metastim.utils import MetaStimUtil
from metastim import visualization as vis
import os
import numpy as np

if __name__ == "__main__":
    
    lead_id  = '6172'
    electrode_list = [1, 0, 0, 0, -1, 0, 0, 0]
    stimulation_amp = 3 # [V]
    pulse_width = 90 #[us]
    num_axons = 10
    min_distance = 1
    max_distance = 5
    axon_diameter = 6 # [um]

    lead_radius = MetaStimUtil.get_lead_radius(lead_id, electrode_list)

    inl = 100 * axon_diameter / 1e3 # distance between nodes on an axon

    z_base = np.arange(-5, 16, inl)
    num_axon_nodes = z_base.shape[0]

    x_axon = np.repeat(np.linspace(min_distance, max_distance, num=num_axons), num_axon_nodes).reshape(num_axon_nodes, num_axons, order='F') + lead_radius
    y_axon = np.zeros(x_axon.shape)
    z_axon = np.repeat(z_base, num_axons).reshape(num_axon_nodes, num_axons)


    axon_ann_model  = axon_ann.AxonANN(electrode_list,  pulse_width, stimulation_amp, num_axons, axon_diameter)
    field_ann_model = field_ann.FieldANN(electrode_list)

    phi_axon = field_ann_model.field_ann(x_axon, y_axon, z_axon)
    (axon_act, ) = axon_ann_model.axon_ann(x_axon, y_axon, z_axon, lead_radius)

    (axon_act, axon_th) = axon_ann_model.axon_ann(x_axon, y_axon, z_axon, lead_radius, threshold=True)

    print(axon_th)

    visual_demo1 = vis.Visualization(lead_id, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_act)
    visual_demo1.visualize(electrode_list)