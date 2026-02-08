import numpy as np
import open3d as o3d
import os

file_name = r"data_1\voxels\frame_000000_voxel.bin"
GRID_SIZE = (64, 32, 64) # x, y, z 유니티에서는 y가 높이

def drawVoxel():
    print(f"파일 읽기 시작 : {file_name}...")
    if not os.path.exists(file_name):
        print("오류 : 파일을 찾을 수 없습니다.")
        return
    
    raw_data = np.fromfile(file_name, dtype=np.uint8).reshape(GRID_SIZE)
    data = np.transpose(raw_data, (0, 2, 1)) # 그림 그릴땐 맨 뒤가 높이
    print("데이터 로드 및 변환 완료")

    occupied_indices = np.argwhere(data != 0)
    if len(occupied_indices) == 0:
        print("경고 : 데이터가 모두 0입니다. (빈맵). 그릴 것이 없습니다.")
        return
    colors = np.zeros((occupied_indices.shape[0], 3))

    for i, (x, y, z) in enumerate(occupied_indices):
        val = data[x, y, z]
        if val == 1: colors[i] = [0.3, 0.3, 0.3] # Road - 회색
        elif val == 2: colors[i] = [0, 0, 1.0] # Car - 파란색
        elif val == 3: colors[i] = [1.0, 0, 0] # Obstacle - 빨간색
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occupied_indices)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)

    print("Open3D 창 열기")

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([voxel_grid, mesh_frame], window_name="My Voxel Map", width=800, height=600)

if __name__ == "__main__":
    drawVoxel()