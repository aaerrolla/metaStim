import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Visualization:
    def __init__(self, lead_radius, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_activation):
        self.lead_radius = lead_radius
        self.stimulation_amp = stimulation_amp
        self.num_axons = num_axons
        self.x_axon = x_axon
        self.z_axon = z_axon
        self.phi_axon = phi_axon
        self.axon_activation = axon_activation

    def visualize(self):
            font = {'family':'serif', 'color':'black', 'size':20}
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            h_lead = self.z_axon[-1,0] - self.z_axon[0,0]
            ax1.add_patch(patches.Rectangle((-self.lead_radius, self.z_axon[0,0]), 2*self.lead_radius, h_lead, linewidth=1, edgecolor='k', facecolor='k'))
            ax1.set_xlim([-1,10])
            ax1.set_ylim([self.z_axon[0,0], self.z_axon[-1,0]])
            for k in range(0, self.num_axons):
                if self.axon_activation[k] > 0:
                    ax1.plot([self.x_axon[0,k], self.x_axon[0,k]], [self.z_axon[0,0], self.z_axon[-1,0]], 'g-', linewidth=1) # blue is active
                else:
                    ax1.plot([self.x_axon[0,k], self.x_axon[0,k]], [self.z_axon[0,0], self.z_axon[-1,0]], 'k-', linewidth=1, alpha=0.25) # black is inactive
                
            ax1.set_title('axons & lead', fontdict = font)
            ax1.set_xlabel('node', fontdict = font)
            ax1.set_ylabel('$\Phi$ (V)', fontdict = font)

            for k in range(0, self.num_axons):
                if self.axon_activation[k] > 0:
                    ax2.plot(self.stimulation_amp * self.phi_axon[:,k], 'g-', linewidth=1) # blue is active
                else:
                    ax2.plot(self.stimulation_amp * self.phi_axon[:,k], 'k-', linewidth=1, alpha=0.25) # black is inactive
                
            ax2.set_title('potentials across axons', fontdict = font)
            ax2.set_xlabel('node', fontdict = font)
            ax2.set_ylabel('$\Phi$ (V)', fontdict = font)
            
            plt.show()
