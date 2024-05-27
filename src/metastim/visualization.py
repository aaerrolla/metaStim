import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as image
import os
import sys

from metastim import lead_selector as ls


class Visualization:
    def __init__(self, lead_id, stimulation_amp, num_axons, x_axon, z_axon, phi_axon, axon_activation):
        lead_selector =  ls.LeadSelector('DBSLead-smry.csv')
        self.leads = lead_selector.load_leads();          
        # self.lead_radius = lead_radius
        self._validate_lead(lead_id)
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

    def _validate_lead(self, lead_id):
        if lead_id not in self.leads.keys():
            print(f"Invalid lead specified. Lead Id must be  of {self.leads.keys()}.")
            sys.exit(7)
        else:
            lead = self.leads.get(lead_id)         
            # get radius 
            self.lead_radius = lead.re

    def visualize1(self, ec):
            font = {'family':'serif', 'color':'black', 'size':20}
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            
            # h_lead = self.z_axon[-1,0] - self.z_axon[0,0]
            # ax1.add_patch(patches.Rectangle((-self.lead_radius, self.z_axon[0,0]), 2*self.lead_radius, h_lead, linewidth=1, edgecolor='k', facecolor='k'))
            # ax1.set_xlim([-1,10])
            # ax1.set_ylim([self.z_axon[0,0], self.z_axon[-1,0]])

            ax1.axis([0, 25, 0, 25])

            # draw controll panel

            x_offset = 5
            y_offset = 5

            cell_labels = ['1', '2', '3', '4', '5', '6', '7', '8']
            x_positions = [1, 1, 3, 5, 1, 3, 5, 1] 
            y_positions = [1, 4, 4, 4, 7, 7, 7, 10]

            for i in range(8):
                color = None
                if ec[i] == 1:
                    color = 'red'
                elif ec[i] == -1:
                    color = 'skyblue'
                elif ec[i] == 0:
                    color = 'gray'

                x = x_positions[i] + x_offset
                y = y_positions[i] + y_offset

                if ( i == 0 or i == 7):        
                    # Row 1 & 4
                    shape = patches.Rectangle((x , y ), width=5.8, height=2, color=color)
                    ax1.add_patch(shape)
                    if i == 0:
                        ax1.text(x + 3, y + 1, cell_labels[i], ha='center', va='center', fontsize=12)
                    if i == 7:
                        ax1.text(x + 3, y + 1, cell_labels[i], ha='center', va='center', fontsize=12)
                else:
                    # Row 2 , 3, 4, 5, 6, 7
                    shape = patches.Rectangle((x, y), width=1.6, height=2, color=color, linewidth=2)
                    ax1.add_patch(shape)
                    ax1.text(x + 1, y + 1, cell_labels[i], ha='center', va='center', fontsize=12)

            # lead image 
            # lead_image = image.imread('./images/lead2.png')
            # ax1.imshow(lead_image)

            # leads             
            for k in range(0, self.num_axons):
                if self.axon_activation[k] > 0:
                    ax1.plot([self.x_axon[0,k] + 16, self.x_axon[0,k] + 16], [self.z_axon[0,0] + 5, self.z_axon[-1,0]], 'g-', linewidth=1) # blue is active
                else:
                    ax1.plot([self.x_axon[0,k] + 16, self.x_axon[0,k] + 16], [self.z_axon[0,0] + 5, self.z_axon[-1,0]], 'k-', linewidth=1, alpha=0.25) # black is inactive
                
            ax1.set_title('axons & lead', fontdict = font)
            ax1.set_xlabel('node', fontdict = font)
            ax1.set_ylabel('$\Phi$ (V)', fontdict = font)

            # Pottentials

            for k in range(0, self.num_axons):
                if self.axon_activation[k] > 0:
                    ax2.plot(self.stimulation_amp * self.phi_axon[:,k], 'g-', linewidth=1) # blue is active
                else:
                    ax2.plot(self.stimulation_amp * self.phi_axon[:,k], 'k-', linewidth=1, alpha=0.25) # black is inactive
                
            ax2.set_title('potentials across axons', fontdict = font)
            ax2.set_xlabel('node', fontdict = font)
            ax2.set_ylabel('$\Phi$ (V)', fontdict = font)
            
            plt.show()
