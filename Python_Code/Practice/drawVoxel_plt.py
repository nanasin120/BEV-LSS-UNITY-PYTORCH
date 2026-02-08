import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

file_name = r"C:\Users\MSI\Desktop\DrivingData\Log_20260127_222102\voxels\frame_000000_voxel.bin"

voxelarray = np.zeros((32, 32, 16), dtype=bool)

data = np.fromfile(file_name, dtype=np.uint8).reshape(32, 16, 32)
data = np.transpose(data, (0, 2, 1))

voxelarray = data != 0

colors = np.empty(shape=(32, 32, 16), dtype=object)
colors[data == 1] = 'black'
colors[data == 2] = 'yellow'
colors[data == 3] = 'red'


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(voxelarray, facecolors=colors, edgecolors='k')
ax.set_box_aspect([32, 32, 16])

plt.show()