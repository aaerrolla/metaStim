import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
num_electrodes = 8  # Number of electrodes in one lead
num_points = 100  # Number of points in each dimension
amplitude = 1.0  # Amplitude of stimulation

# Create a grid of points representing the brain tissue
x = np.linspace(-1, 1, num_points)
y = np.linspace(-1, 1, num_points)
X, Y = np.meshgrid(x, y)

# Calculate the field activation and axon activation at each point
field_activation = np.zeros_like(X)
axon_activation = np.zeros_like(X)
for i in range(num_electrodes):
    # Generate random voltage input for each electrode
    voltage_input = np.random.choice([-1, 0, 1])  # Negative, zero, or positive voltage
    
    # Generate random position for each electrode
    lead_x = np.random.uniform(-1, 1)
    lead_y = np.random.uniform(-1, 1)
    
    # Calculate the distance between each point and the electrode
    distance = np.sqrt((X - lead_x)**2 + (Y - lead_y)**2)
    
    # Add the contribution of this electrode to the field activation
    field_activation += voltage_input * amplitude / (distance + 0.1)  # Adding 0.1 to avoid division by zero
    
    # Add the contribution of this electrode to the axon activation
    axon_activation += voltage_input * amplitude * np.exp(-distance**2)  # Gaussian decay function

# Plot the field activation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, field_activation, cmap='coolwarm')
plt.colorbar(label='Field Activation')
plt.title('Field Activation in Deep Brain Stimulation')
plt.xlabel('X')
plt.ylabel('Y')

# Plot the axon activation
plt.subplot(1, 2, 2)
plt.contourf(X, Y, axon_activation, cmap='coolwarm')
plt.colorbar(label='Axon Activation')
plt.title('Axon Activation in Deep Brain Stimulation')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
