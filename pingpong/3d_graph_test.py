import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mnist_3 import two_layer_model

# Assuming you have your data in this format
layer1_input = np.array([8, 16, 32, 64, 128, 256])
layer2_input = np.array([8, 16, 32, 64, 128, 256])


accuracies = np.empty(shape=(layer1_input.size, layer2_input.size))

print("DUPPPPPAAAAA")

for inx1, i1 in enumerate(layer1_input):
    for inx2, i2 in enumerate(layer2_input):
        print("DUPA: ", i1, i2, "\n\n")
        accuracies[inx1][inx2] = two_layer_model(i1, i2, 'relu')

# Create a meshgrid for interpolation
x_fine = np.linspace(layer1_input.min(), layer1_input.max(), 100)
y_fine = np.linspace(layer2_input.min(), layer2_input.max(), 100)
X, Y = np.meshgrid(x_fine, y_fine)

# Prepare data for interpolation
x = np.repeat(layer1_input, layer1_input.size)
y = np.tile(layer2_input, layer2_input.size)
z = accuracies.flatten()

# Interpolate
Z = griddata((x, y), z, (X, Y), method='cubic')

# Create a figure with two subplots
fig = plt.figure(figsize=(16, 6))

# Heatmap
ax1 = fig.add_subplot(121)
im = ax1.imshow(Z, extent=[layer1_input.min(), layer1_input.max(), layer2_input.min(), layer2_input.max()],
                origin='lower', aspect='auto', cmap='viridis')
ax1.set_title('Accuracy Heatmap')
ax1.set_xlabel('Size of 1st layer')
ax1.set_ylabel('Size of 2nd layer')
plt.colorbar(im, ax=ax1, label='Accuracy')

# 3D Surface plot
ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax2.set_title('Accuracy 3D Surface')
ax2.set_xlabel('Size of 1st layer')
ax2.set_ylabel('Size of 2nd layer')
ax2.set_zlabel('Accuracy')
fig.colorbar(surf, ax=ax2, label='Accuracy')

plt.tight_layout()
fig.savefig("fig.png")
plt.show()

header = 'Layer 1,' + ','.join(map(str, layer2_input))

savedata = np.column_stack((layer1_input, accuracies))

np.savetxt('output_2d.csv', savedata, delimiter=',', header = header, comments='')
