import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LSS
import open3d as o3d
import numpy as np
from UnityDataset import UnityDataset
import random

def drawVoxel(raw_data):
    print(raw_data.shape)
    data = np.transpose(raw_data, (1, 0, 2)) # 높이, 앞뒤, 좌우 로 변경
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSS(device).to(device=device)
model.load_state_dict(torch.load(r'model_save\model_epoch_55.pth', weights_only=True))
model.eval()

root_dir = [r'./data_1', r'./data_2', r'./data_3', r'./data_4']
full_dataset = UnityDataset(root_dir)
sample_idx = random.randint(0, len(full_dataset))
sample = full_dataset[sample_idx]

imgs = sample['imgs'].unsqueeze(0).to(device)
rots = sample['rots'].unsqueeze(0).to(device)
trans = sample['trans'].unsqueeze(0).to(device)
intrins = sample['intrinsics'].unsqueeze(0).to(device)
label = sample['label_3d'] # 1, 32, 64, 64 높이, 좌우, 앞뒤

print(imgs.shape)
print(rots.shape)
print(trans.shape)
print(intrins.shape)

with torch.no_grad():
    preds = model(imgs, rots, trans, intrins)
    pred_voxel = torch.argmax(preds[0], dim=0).cpu().numpy() # 32, 64, 64
    # print(pred_voxel)

    drawVoxel(label.squeeze(0).cpu().numpy()) # 좌우, 높이, 앞뒤 로 변경
    drawVoxel(pred_voxel) # 좌우, 높이, 앞뒤 로 변경