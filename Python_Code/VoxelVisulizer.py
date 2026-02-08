import open3d as o3d
import numpy as np

class VoxelVisualizer:
    def __init__(self, window_name = "Voxel"):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=800, height=600)
        
        self.pcd = o3d.geometry.PointCloud()
        self.voxel_grid = None
        self.is_first = True
        
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)

    def update(self, raw_data):
        data = np.transpose(raw_data, (1, 0, 2)) 
        occupied_indices = np.argwhere(data != 0)
        
        if len(occupied_indices) == 0:
            print("What the Hack")
            return

        colors = np.zeros((occupied_indices.shape[0], 3))
        for i, (x, y, z) in enumerate(occupied_indices):
            val = data[x, y, z]
            if val == 1: colors[i] = [0.3, 0.3, 0.3] # Road
            elif val == 2: colors[i] = [0, 0, 1.0]   # Car
            elif val == 3: colors[i] = [1.0, 0, 0]   # Obstacle

        self.pcd.points = o3d.utility.Vector3dVector(occupied_indices)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        if self.voxel_grid is not None:
            self.vis.remove_geometry(self.voxel_grid, reset_bounding_box=False)

        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=1.0)
        
        if self.is_first:
            self.vis.add_geometry(self.voxel_grid, reset_bounding_box=True) 
            self.is_first = False
        else:
            self.vis.add_geometry(self.voxel_grid, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()