# MetaStim DBS in Python

### How to use metastim  python package

#### Install Instructions:

1. create python3 virtual environment  (  python version  required  3.8   or higher)


```    
mkdir  -p  ~/project/metastim  # or any directory  of  your choice
cd project/metastim
python3 -m venv venv
```

2. Activate the virtual environment 

```
source venv/bin/activate
```
if above command is scuccess  your terminal prompt will change to denote that  you are in python virtual environment.


3. Install metastim package

from being in python virtual environment  type 

```
pip install metastim
```

4. start python interactive shell

```
python3  
```

5. Test if you can import metastim modules 

try to import metastim modules filed_ann and axon_ann  as shown below 
if you do not see any errors ,  you successfully installed metastim package 

```
>>> from metastim import field_ann, axon_ann
```


#### How to Use MetaStim 

Sample demo code is provided below, it shows usage of metastim 
copy this code into let say demo.py  in  ~/project/metastim  folder 
and run within virtual environment created above 

or 

alternatively you can download [demo.py](./demo.py) avialable in this repositoty and run it in the virtual environment created above

```
python3 demo.py
```

Here is the complete code.

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

1. donwload demo.ipynb  and copy to  ~/projects/metastim folder 

2. activate python virtual environment if it is not activated already 

```
cd ~/projects/metastim

source venv/bin/activate
```

3. Open  vscode 

```
code .
```

4. Run demo.ipynb jupitor notebook file 

After vscode opens, open demo.ipynb file and click "Run All" button
vscode suggests to install  ipython server if ipython is not already installed 
install ipython by repsonding to vscode prompt.

If ipython server already present in vscode then vscode ask to select environment , select the virtual env 
