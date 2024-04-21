# metaStim
DBS Stimulation

## python requirements 

python version  3.9 or higher 

## install required packages 

pip install -r requirements.txt 

## packages and classes

packages 

1. utils.py
2. axon_ann.py
3. visualization.py



utils.py :  contains MetaStimUtil class 
            MetaStimUtil class contains utility methods that are used by axon_ann.py 

            get_field_sd() : returns 11 second differences of electric potentials per axon
            get_field_shape() : shape classification for 11SD for each axon 
            get_axon_to_lead_dist(): distance from lead to axon

axon_ann.py: contains class AxonANNModel which contains methods to genrate the axonn coordinates
             electric potentials and axon activation.

             _init__() class initializer takes 8 input parameters              
             electrode_list, lead_radius, num_axons, min_distance, max_distance, axon_diameter, pulse_width , stimulation_amp  and contructs instance of this class

             axon_coord() : generates and returns x, y, and z coordinates of axon
             field_ann() : returns electric potentials from Field ANN for each axon
             axon_ann():  returns axon activation based on electric potentials
             validate_input(): validates the input values for axon_diameter, pulse_width

visualization.py:  contains matplotlib function to visulize the stimulation 

                    __init__():  initializer which takes all required parameters to generate the plot
                                 parameters are lead_radius, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_activation
                    visualize() : generates the plot


## flow
    1. construct  AxonANNModel  instance by passing all initilization parameters 
    2. call axon_coord() method on the AxonANNModel instance , which  retuns axon coordinates  (x_axon, y_axon, z_axon)
    3. call field_ann() method  on the AxonANNModel instance , which  returns  electric potentials (phi_axon)
    4. call axon_ann() method on  the AxonANNModel instance , which  returns axon activation (axon_activation)
    5. construct  Visualization instance by passing 
        lead_radius, stimulation_amp, num_axons ( inputs )
        and  x_axon, z_axon, phi_axon, axon_act ( are genrated in previous steps)
    6. call visualize() metohd to generate plot

#### Initial Review Comments

