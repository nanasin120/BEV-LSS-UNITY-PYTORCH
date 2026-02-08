import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LSS

if __name__ == "__main__":
    # 1. 모델 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LSS(device).to(device)

    # 2. 가짜 데이터 만들기 (Batch=1, Cam=6, C=3, H=128, W=352)
    dummy_imgs = torch.randn(1, 6, 3, 128, 352).to(device)
    dummy_rots = torch.eye(3).view(1, 1, 3, 3).repeat(1, 6, 1, 1).to(device)
    dummy_trans = torch.zeros(1, 6, 3).to(device)
    dummy_intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, 6, 1, 1).to(device)
    
    # 외부행렬 합치기 (4x4)
    dummy_extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 6, 1, 1).to(device)
    dummy_extrinsics[..., :3, :3] = dummy_rots
    dummy_extrinsics[..., :3, 3] = dummy_trans

    # 3. 실행!
    print("모델 실행 중...")
    output = model(dummy_imgs, dummy_rots, dummy_trans, dummy_intrinsics)
    
    print("성공! 🎉")
    print("Output Shape:", output.shape) 
    # 예상 결과: torch.Size([1, 3, 32, 64]) 혹은 설정에 따라 (1, 3, 64, 64)