# MetaStim DBS in Python

### Instructions  to  use metastim  python package

#### Install Instructions:

1. create python3 virtual environment  (  python version  required  3.8   or higher)

```    
python3 -m venv venv
```

2. Activate the virtual environment 

```
source venv/bin/activate
```
3. Install metastim package

```
pip install metastim
```



#### Usage Example 

Below provided example demo on  how to use metastim 
copy this code into let say  demo.py  and  run  from the same virtual environment created above 

```
python3 demo.py
```

Here is the complete 

```
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


    axon_ann_model = axon_ann.AxonANN(electrode_list,  pulse_width, stimulation_amp, num_axons, axon_diameter)
    field_ann_model = field_ann.FieldANN(electrode_list)

    phi_axon = field_ann_model.field_ann(x_axon, y_axon, z_axon)
    axon_act = axon_ann_model.axon_ann(x_axon, y_axon, z_axon, lead_radius)

    visual_demo1 = vis.Visualization(lead_id, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_act)
    visual_demo1.visualize1(electrode_list)

```

#### Sample Jupitor Notebook example 


A Sample Jupitor note book file is aviable in this repository  at  [demo.ipynb](./demo.ipynb)







