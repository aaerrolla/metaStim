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

    def visualize1(self):
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

    def visualize2(self, ec):
            font = {'family':'serif', 'color':'black', 'size':20}
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            
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

            # draw lead image 

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


    def visualize(self, ec):
            font = {'family':'serif', 'color':'black', 'size':20}
            
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            
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
            ax1.set_title('Control Panel', fontdict = font)

            # draw lead image 
            
            ax2.set_xlim(0, 2)
            ax2.set_ylim(0, 10)
            ax2.set_aspect('equal')


            main_body_width = 0.3
            main_body_height = 6
            main_body = patches.Rectangle((0.85, 2), main_body_width, main_body_height, linewidth=1, edgecolor='gray', facecolor='white')
            ax2.add_patch(main_body)


            rect_height = 1
            spacing = 0.5
            gray_rects = [
                (0.85, 6.5),  # First rectangle at the top
                (0.85, 5),    # Second rectangle
                (0.85, 3.5),  # Third rectangle
                (0.85, 2)     # Fourth rectangle
            ]

            for rect in gray_rects:
                if rect[1] == 5 or rect[1] == 3.5:
                    cylinder_width = main_body_width / 3
                    ax2.add_patch(patches.Rectangle((rect[0], rect[1]), cylinder_width, rect_height, linewidth=1, edgecolor='gray', facecolor='gray'))
                    ax2.add_patch(patches.Rectangle((rect[0] + 2 * cylinder_width, rect[1]), cylinder_width, rect_height, linewidth=1, edgecolor='gray', facecolor='gray'))
                elif rect[1] == 2:
                    ax2.add_patch(patches.Rectangle((rect[0], rect[1]), main_body_width, rect_height / 1.5, linewidth=1, edgecolor='gray', facecolor='gray'))
                else:
                    ax2.add_patch(patches.Rectangle((rect[0], rect[1]), main_body_width, rect_height , linewidth=1, edgecolor='gray', facecolor='gray'))

            
            ellipse_width = main_body_width
            ellipse_height = 0.35
            ellipse = patches.Ellipse((1.0052, 2), ellipse_width, ellipse_height, angle=0, linewidth=1, edgecolor='gray', facecolor='white')
            ax2.add_patch(ellipse)

            
            bottom_black = patches.Ellipse((1.0052, 2), ellipse_width, ellipse_height, angle=0, linewidth=1, edgecolor='gray', facecolor='gray')
            ax2.add_patch(bottom_black)
            ax2.set_title('lead', fontdict = font)
            
            
            # Plot Pottentials

            for k in range(0, self.num_axons):
                if self.axon_activation[k] > 0:
                    ax3.plot(self.stimulation_amp * self.phi_axon[:,k], 'g-', linewidth=1) # blue is active
                else:
                    ax3.plot(self.stimulation_amp * self.phi_axon[:,k], 'k-', linewidth=1, alpha=0.25) # black is inactive
                
            ax3.set_title('potentials across axons', fontdict = font)
            ax3.set_xlabel('node', fontdict = font)
            ax3.set_ylabel('$\Phi$ (V)', fontdict = font)
            
            plt.show()
