import numpy as np


class MetaStimUtil():

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
    @staticmethod
    def get_field_sd(num_axons, phi_axon):
        sd_axon = 1e3 * np.diff(phi_axon, n=2, axis=0) # V => mV
        sd_max_indx = np.argmax(sd_axon, axis=0) # row index per column where SD is max

        window_size = 11 # window size
        nn = int((window_size - 1) / 2) # number of neighbors to left/right of max
        sd_11_axon = np.zeros((window_size, sd_axon.shape[1])) # pre-allocate 11SD for each column

        # define the bounds of the window
        w_indx_l = np.maximum(sd_max_indx - nn, np.zeros(sd_max_indx.shape, dtype=int)) # first index of window
        w_indx_r = np.minimum(sd_max_indx + nn, (sd_axon.shape[0] - 1) * np.ones(sd_max_indx.shape, dtype=int)) # last index of window

        # in window is < 11 values, get pad counts
        pad_l = np.maximum(nn - sd_max_indx, np.zeros(sd_max_indx.shape, dtype=int)) # how many to zero-pad to the left
        pad_r = np.maximum(sd_max_indx - sd_axon.shape[0] + 1, np.zeros(sd_max_indx.shape, dtype=int)) # ditto for the right

        for k in range(0, num_axons):
            sd_11_axon[:,k] = np.pad(sd_axon[w_indx_l[k]:w_indx_r[k]+1, k], (pad_l[k], pad_r[k]), 'constant')
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
    @staticmethod
    def get_field_shape(num_axons, sd_11_axon):
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
    @staticmethod
    def get_axon_to_lead_dist(lead_radius, x_axon, y_axon):
        # calculate distance of axon to z axis at xy = (0,0)
        axon_radius = np.sqrt(np.min(x_axon, axis=0) ** 2 + np.min(y_axon, axis=0) ** 2)
        axon_distance = axon_radius - lead_radius
        axon_distance[axon_distance < 0] = 'NaN'
        return axon_distance
