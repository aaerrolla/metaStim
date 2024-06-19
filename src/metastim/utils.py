import numpy as np
import sys
from metastim import lead_selector


class MetaStimUtil:

    # FUNCTION = field_sd(phi (req))
    # Short description: this is an auxiliary function that calculates SD from phi
    # req = required
    # SD = second (discrete) difference operation
    # INPUTS:
    # phi, electric potentials [in V] in a # points x # axon array by format
    # OUTPUT:
    # SD, 11 second differences of electric potentials per axon [in mV]
    # (format: 11 x # axons)
    # NOTES:
    # - each axon in this demo has the same number of nodes / points
    # - the code should be generalized so that each axon can have different # pts / axon
    # - however, by this part in the code, only 11SD are calculated per axon, so data can be held in a 2D array    
    @classmethod
    def get_field_sd(cls, num_axons, phi_axon):
        sd_axon = 1e3 * np.diff(phi_axon, n=2, axis=0)  # V => mV
        sd_max_indx = np.argmax(sd_axon, axis=0)  # row index per column where SD is max

        window_size = 11  # window size
        nn = int((window_size - 1) / 2)  # number of neighbors to left/right of max
        sd_11_axon = np.zeros((window_size, sd_axon.shape[1]))  # pre-allocate 11SD for each column

        for k in range(0, num_axons):
            # define the bounds of the window
            w_indx_l = np.maximum(sd_max_indx[k] - nn, 0)  # first index of window
            w_indx_r = np.minimum(sd_max_indx[k] + nn, sd_axon.shape[0] - 1)  # last index of window

            # calculate padding
            pad_l = nn - (sd_max_indx[k] - w_indx_l)
            pad_r = nn - (w_indx_r - sd_max_indx[k])

            # pad the values within the window
            sd_11_axon[:, k] = np.pad(sd_axon[w_indx_l:w_indx_r + 1, k], (pad_l, pad_r), 'constant')

        return sd_11_axon

    # FUNCTION = field_shape(SD (req))
    # Short description: this is an auxiliary function that assigns a shape classification of potentials for each axon
    # req = required
    # INPUT:
    # SD, 11 second differences of electric potentials per axon [in mV]
    # (format: 11 x # axons)
    # OUTPUT:
    # Fields Shape (FS), shape classification for 11SD for each axon (1 x # axons)
    # FS can be 1, 2, or 3
    # NOTES:
    # -this is a stopgap based on a simple thresholding of a feature
    # -a more advanced classification will be implemented in a future code ver
    @classmethod
    def get_field_shape(cls, num_axons, sd_11_axon):
        fs_axon = np.zeros((num_axons,))

        # calculate maximum SD for each axon
        # calculate minimum SD for each axon
        # calculate absolute value of ratio of max SD / min SD
        sd_max = np.max(sd_11_axon, axis=0)
        sd_min = np.min(sd_11_axon, axis=0)
        sd_rat = np.abs(np.divide(sd_max, sd_min))
        fs_axon[sd_rat >= 2.55] = 1
        fs_axon[(sd_rat >= 1.15) & (sd_rat <2.55)] = 2
        fs_axon[sd_rat < 1.15] = 3
        return fs_axon


    # FUNCTION = axon2lead_dist(lead (rew), axonCoord (req))
    # Short description: this is an auxiliary function that calculates the distance of each axon to the lead
    # req = required
    # INPUT:
    # lead, lead geometry
    # -for this demo, we only need the lead's radius
    # axon_coord, coordinates of axon 
    # OUTPUT:
    # d, distance from lead to axon (# axons x 1)
    # NOTES:
    # -this calculation works for an ideal setup where the lead and axons are perfectly parallel to each other and aligned with the z axis
    # -future version of this code will calculate d for arbitrary lead angles and non-straight axon trajectories
    @classmethod
    def get_axon_to_lead_dist(cls, lead_radius, x_axon, y_axon):
        # calculate distance of axon to z axis at xy = (0,0)
        axon_radius = np.sqrt(np.min(x_axon, axis=0) ** 2 + np.min(y_axon, axis=0) ** 2)
        axon_distance = axon_radius - lead_radius
        axon_distance[axon_distance < 0] = 'NaN'
        return axon_distance


    @classmethod
    def get_lead_radius(cls, lead_id, electrode_list):
        leadselector =  lead_selector.LeadSelector('DBSLead-smry.csv')
        leads = leadselector.load_leads();        
        
        if lead_id not in leads.keys():
            print(f"Invalid lead specified. Lead Id must be  of {leads.keys()}.")
            sys.exit(5)
        else:
            lead = leads.get(lead_id)
            if lead.no != len(electrode_list) :
                print(f"Invalid electrode configuration. {lead_id} contains {lead.no} electrods")
                sys.exit(6)

            # get radius 
            return lead.re