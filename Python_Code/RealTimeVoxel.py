from mlagents_envs.environment import UnityEnvironment
import numpy as np
import open3d as o3d
import torch
from model import LSS
from VoxelVisulizer import VoxelVisualizer

env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LSS(device).to(device)
model.load_state_dict(torch.load(r'model_save\model_epoch_30.pth', weights_only=True))
model.eval()

pred_visualizer = VoxelVisualizer("pred")
# real_visualizer = VoxelVisualizer("real")

print(f"연결 성공! Behavior Name: {behavior_name}")

try:
    while True:
        decision_steps, _ = env.get_steps(behavior_name)

        if len(decision_steps) > 0:
            # for i, obs_spec in enumerate(spec.observation_specs):
            #     print(f'Index {i}: {obs_spec.name} Shape : {obs_spec.shape}')

            imgs = []
            for i in range(6):
                img = torch.from_numpy(decision_steps.obs[i]).float() # 1, 128, 352, 3
                img = img.permute(0, 3, 1, 2) # 1, 3, 128, 352
                imgs.append(img)
            imgs = torch.stack(imgs, dim=1).to(device) # # 1, 6, 3, 128, 352

            datas = torch.from_numpy(decision_steps.obs[6]).to(device) # 1, 120 + 131072

            # print(datas.shape)
            datas = datas.squeeze(0) # 120 + 131072
            # voxel = datas[120:] # 131072
            # voxel = voxel.reshape(64, 32, 64) # 좌우, 상하, 앞뒤
            # voxel = voxel.permute(1, 0, 2).cpu().numpy() # 상하, 좌우, 앞뒤
            arr = datas[:120] # 120
            arr = arr.reshape(6, 20) # 6, 20

            rots, trans, intrins = [], [], []

            for i in range(6):
                data = arr[i]
                r = torch.tensor([[data[0],  data[1],  data[2]],
                        [ data[4],  data[5],  data[6]],
                        [ data[8], data[9], data[10]]], device=device)

                t = torch.tensor([data[3], data[7], data[11]], device=device)

                intrin = torch.tensor([[data[16], 0,          data[18]],
                    [ 0,        data[17],   data[19]],
                    [ 0,        0,          1]], device=device)
                
                rots.append(r)
                trans.append(t)
                intrins.append(intrin)

            rots = torch.stack(rots, dim=0).unsqueeze(0).to(device)
            trans = torch.stack(trans, dim=0).unsqueeze(0).to(device)
            intrins = torch.stack(intrins, dim=0).unsqueeze(0).to(device)

            print(f'imgs : {imgs.shape} | rots : {rots.shape} | trans : {trans.shape} | intrins : {intrins.shape}')
            
            preds = model(imgs, rots, trans, intrins)
            pred_voxel = torch.argmax(preds[0], dim=0).cpu().numpy()

            pred_visualizer.update(pred_voxel)
            # real_visualizer.update(voxel)
            
        env.step()
finally:
    env.close()