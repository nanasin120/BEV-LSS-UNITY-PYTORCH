import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18

# efficient-B0 사용 이유
# 제한된 자원환경에서 깊이, 너비, 해상도를 비례적으로 확장하는 방식으로 철저한 아키텍처 탐색을 통해 발견된 네트워크 구조이다.
# 이미지를 농축해서 반환함
class ImageBackbone(nn.Module):
    def __init__(self):
        super(ImageBackbone, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) # 만들어진 efficientnet_b0 가져옴
        self.backbone = self.model.features[:-1] # 맨 마지막은 제외, 맨 마지막을 320에서 1280으로 만듬

    def forward(self, x):
        # x : [Batch, N, 3, 128, 352]
        B, N, C, H, W = x.shape

        # EfficientNet에 넣기 위해 5차원을 4차원으로 줄여줌
        # [B * N, 3, 128, 352]
        x = x.view(B * N, C, H, W)

        # 채널이 112가 됨
        # [B * N, 112, 8, 22]
        x_skip = x
        for i in range(6): x_skip = self.backbone[i](x_skip)

        # 채널이 320가 됨
        # [B * N, 320, 4, 11]
        x_final = x_skip
        for i in range(6, len(self.backbone)): x_final = self.backbone[i](x_final) # 채널이 320가 됨

        # B * N을 했으니 다시 되돌려줌
        # FC, FH, FW = 112, 8, 22
        FC, FH, FW = x_skip.shape[1], x_skip.shape[2], x_skip.shape[3]
        # [B, N, 112, 8, 22]
        x_skip = x_skip.view(B, N, FC, FH, FW)

        # FC, FH, FW = 320, 4, 11
        # [B, N, 320, 4, 11]
        FC, FH, FW = x_final.shape[1], x_final.shape[2], x_final.shape[3]
        x_final = x_final.view(B, N, FC, FH, FW)

        # x_final : 정보를 가장 응축해놓은 것
        # x_skip : 그보다 위의, 구체적 위치, 형태 등의 정보가 남아있음
        return x_final, x_skip
    
# 고해상도 [B, N, 512, 8, 22] 짜리 데이터 생성
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Up, self).__init__()

        # scale_factor : Width, Height에 그만큼 곱해준다.
        # bilinear : 선형 보간법으로 채워준다.
        # align_corners : 양 끝을 고정한다.
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        # 최종적으로는 (입력 + 2 - 3)/1+1 = 입력이어서 변화는 없다.
        # 채널만 변화한다.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # 연산이 가능하게 맞추어줌
        B1, N1, C1, H1, W1 = x1.shape # [B, N, 320, 4, 11] x_final
        B2, N2, C2, H2, W2 = x2.shape # [B, N, 112, 8, 22] x_skip
        x1 = x1.view(B1 * N1, C1, H1, W1) # [B * N, 320, 4, 11]
        x2 = x2.view(B2 * N2, C2, H2, W2) # [B * N, 112, 8, 22]

        # x1은 EfficientNet에 더 깊이 들어갔기 때문에 Width와 Height가 작음
        # 그래서 upSampling해줌
        # x1 : [B * N, 320, 8, 22]
        x1 = self.up(x1)

        # 둘을 붙여줌
        # x1 : [B * N, 432, 8, 22]
        x1 = torch.cat([x2, x1], dim=1)

        # 연산해줌
        # x_out : [B * N, 512, 8, 22] 여기서 512는 처음 LSS에서 self.up1을 만들떄 설정한 값임
        x_out = self.conv(x1)

        # 나온걸 다시 되돌려줌
        _, C, H, W = x_out.shape
        # [B, N, 512, 8, 22]
        x_out = x_out.view(B1, N1, C, H, W)
        return x_out
    
# 깊이 [B, N, 45, 8, 22], 특징 [B, N, 64, 8, 22]를 반환
class depthNet(nn.Module):
    def __init__(self, D, C):
        super(depthNet, self).__init__()
        self.D = D # 깊이 45
        self.C = C # 특징 64
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def forward(self, x):
        # x는 [B, N, 512, 8, 22]
        B, N, C, H, W = x.shape
        # x : [B * N, 512, 8, 22]
        x = x.view(B * N, C, H, W)
        
        # x : [B * N, 45 + 64, 8, 22]
        x = self.depthnet(x)

        depth = x[:, :self.D] # depth : [B * N, 45, 8, 22]
        depth = F.softmax(depth, dim=1) # 깊이를 기준으로 확률분포를 생성, 어느 깊이가 가장 가능성 있는 깊이인지 알 수 있음

        feature = x[:, self.D:] # featrue : [B * N, 64, 8, 22]

        depth = depth.view(B, N, self.D, H, W) # depth : [B, N, 45, 8, 22]
        feature = feature.view(B, N, self.C, H, W) # featrue : [B, N, 64, 8, 22]

        return depth, feature

class secondBackbone(nn.Module):
    def __init__(self, in_channel = 64):
        super(secondBackbone, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channel),
            nn.ReLU()            
        )

    def forward(self, x):
        # x : [B, C, D, H, W]
        # x_out : [B, C, D, H, W]
        x_out = self.layers(x)

        return x_out + x # 성능 보존

class TaskHead(nn.Module):
    def __init__(self, in_channel=64, out_channel=4):
        super(TaskHead, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        
    def forward(self, x):
        out = self.conv(x)

        return out

class LSS(nn.Module):
    def __init__(self, device):
        super(LSS, self).__init__()
        self.D = 45 # 깊이
        self.C = 64 # 특징
        self.final_H = 8 # frustum의 height
        self.final_W = 22 # frustum의 width
        self.D_min = 1 # 깊이 최소값
        self.D_max = 46 # 깊이 최댓값+1, 즉 최댓값은 45
        self.D_step = 1 # 깊이 증가값, 1이니 1, 2, 3, ... , 44, 45 이렇게 됨
        self.x = 16 # 복셀 좌우
        self.y = 16 # 복셀 앞뒤
        self.z = 8 # 복셀 상하

        self.device = device # cuda or cpu

        self.imageBackbone = ImageBackbone()
        self.up1 = Up(320 + 112, 512)
        self.depthnet = depthNet(self.D, self.C)
        self.secondBackbone = secondBackbone()
        self.taskHead = TaskHead()

        self.register_buffer('frustum', self.create_frustum())

    def forward(self, x, rots, trans, intrinsics):
        # x : [N, N, 3, 128, 352] 이미지가 들어있음
        # rots : [B, N, 3, 3] 회전값이 들어있음
        # trans : [B, N, 3] 자동차원점을 기준으로 어디에 위치하는지 들어있음
        # intrinsics : [B, N, 3, 3] 카메라 렌즈와 센서의 특성이 들어있음

        # x_final : (B, N, 320, 4, 11) 이미지 데이터 굉장히 농축됨
        # x_skip : (B, N, 112, 8, 22) 이미지 데이터 덜 농축됨, 그래서 형태가 남아있음
        x_final, x_skip = self.imageBackbone(x)

        # x_up : [B, N, 512, 8, 22]
        # x_final과 x_skip을 합쳐 형태가 남아있는 굉장히 농축된 데이터가 됨
        x_up = self.up1(x_final, x_skip)

        # depth : [B, N, 45, 8, 22] 깊이에 대한 값이 확률 분포로 나타나있음
        # featrue : [B, N, 64, 8, 22] 특징 64개가 있음
        depth, feature = self.depthnet(x_up)

        # depth.unsqueeze(3) : [B, N, 45, 1, 8, 22]
        # feature.unsqueeze(2) : [B, N, 1, 64, 8, 22]
        # contextVector : [B, N, 45, 64, 8, 22]
        # 깊이와 특징을 둘 다 가진 contextVector 가완성됨
        contextVector = depth.unsqueeze(3) * feature.unsqueeze(2)

        # geometry : [B, N, 45, 8, 22, 3]
        # 0 : 좌우, 1 : 상하, 2 : 앞뒤
        geometry = self.get_geometry(rots, trans, intrinsics)

        # 미완성 복셀맵이 나옴 [B, 64, 32, 64, 64] 상하, 좌우, 앞뒤
        voxel_features = self.voxel_pooling(geometry, contextVector)

        # [B, 64, 32, 64, 64] 더 관계가 강화되어서 나옴 [상하, 좌우, 앞뒤]
        voxel = self.secondBackbone(voxel_features)

        # [B, 4, 32, 64, 64] 상하, 좌우, 앞뒤
        head = self.taskHead(voxel)

        return head

    def create_frustum(self):
        # ogfH, ogfW는 격자 블록의 크기 [8, 22]
        # Output Grid Final Height/Width
        # 지금까지 만든 [8, 22]의 평면을 3차원으로 Shoot할 발사대를 만들거임
        ogfH, ogfW = self.final_H, self.final_W 

        full_W, full_H = 352, 128

        # 0 ~ ogfW-1까지 ogfW개의 1차원 행렬 생성
        # ogfW 22이니 xs = [0, 1, 2, 3, ... , 20, 21]   
        xs = torch.linspace(0, full_W - 1, ogfW).float().to(self.device)

        # 0 ~ ogfH-1까지 ogfH개의 1차원 행렬 생성
        # ogfH는 8이니 ys = [0, 1, 2, 3, 4, 5, 6, 7]
        ys = torch.linspace(0, full_H - 1, ogfH).float().to(self.device) 

        # self.D_min부터 self.D_max까지 self.D_step만큼 증가하는 1차원 행렬 생성
        # 1, 46, 1이니 ds = [1, 2, 3 ... , 43, 44, 45]
        ds = torch.arange(self.D_min, self.D_max, self.D_step).float().to(self.device)

        # (ds, ys, ws)인 3차원 행렬 생성, d에는 ds만 적혀있고 y에는 ys만 적혀있고 x에는 xs만 적혀있다.
        # [45, 8, 22]인 3차원 행렬이 생성되는것
        # [깊이, 높이, 너비]인 상태로
        # d에는 깊이, y에는 높이, x에는 너비가 중심으로 들어가게됨
        # d[34, 6, 2] = 35, y[34, 6, 2] = 6, x[34, 6, 2] = 2
        d, y, x = torch.meshgrid(ds, ys, xs, indexing='ij') 

        # 맨 뒤에 3개를 연속으로 쌓아 [45, 8, 22, 3]의 4차원 행렬 생성
        # 이제 [45, 8, 22]를 보면 3개의 x, y, d값을 볼 수 있음
        # x와 y에 d를 곱하는 이유는 깊이에 따라 좌표도 함께 멀어지기 때문이다.
        frustum = torch.stack([x * d, y * d, d], dim=-1)

        return frustum
    
    def get_geometry(self, rots, trans, intrinsics):
        B, N = intrinsics.shape[0], intrinsics.shape[1] # 배치와 이미지개수
        frustum = self.frustum # [45, 8, 22, 3] = [깊이, 높이, 너비, xyz]

        inv_intrinsics = torch.inverse(intrinsics) # 내부 행렬 뒤집기 inv_intrinsics : [B, N, 3, 3]

        # 핀홀 카메라 모델 공식
        # 이 식을 통해 2차원의 이미지 평면을 3차원 공간의 점으로 역투영한다.
        # inv_intrinsics.view(B, N, 1, 1, 1, 3, 3) : [B, N, 1, 1, 1, 3, 3]
        # frustum.view(1, 1, 45, 8, 22, 3, 1) :      [1, 1, 45, 8, 22, 3, 1]
        # [B, N, | 1,  1, 1,  | 3, 3]
        # [1, 1, | 45, 8, 22, | 3, 1]
        # 이렇게 나눠서 보면 B, N이 복사되고, 깊이, 높이, 너비가 복사되고, [역내부행렬]이랑 [xd, yd, d]의 곱인걸 알 수 있음
        points_c = torch.matmul(inv_intrinsics.view(B, N, 1, 1, 1, 3, 3), frustum.view(1, 1, 45, 8, 22, 3, 1))
        # points_c : [B, N, 45, 8, 22, 3, 1]
        # points_c는 마지막 [3, 1]이 카메라 렌즈 중심에서 가로로 얼마나 떨어져 있나, 세로로 얼마나 떨어져 있나, 앞으로 얼마나 떨어져 있나 를 가짐

        # 강체 변환 공식
        # 이 식을 통해 이미지는 외부의 상황에 맞게 회전한다.
        # rots는 [B, N, 3, 3]
        # [B, N, 45, 8, 22, 3, 1]랑 연산해야함
        # [B, N, 1,  1, 1,  3, 3]
        points_w = torch.matmul(rots.view(B, N, 1, 1, 1, 3, 3), points_c)
        # [B, N, 45, 8, 22, 3, 1]이 나옴 
        # 이제 회전행렬을 곱했기 때문에 자동차 정면을 기준으로 회전한 값이 됨

        # trans는 [B, N, 3]
        # [B, N, 45, 8, 22, 3, 1]
        # [B, N, 1,  1, 1,  3, 1]
        points_w = points_w + trans.view(B, N, 1, 1, 1, 3, 1)
        # 이제 원래 카메라의 위치도 더해짐

        # 이제 [B, N, 45, 8, 22, 3, 1]를 [B, N, 45, 8, 22, 3]로 만들어준다.
        # N번 카메라의 (깊이, 높이, 너비)는 자동차 정면을 기준으로 회전한 x, y, z를 갖게 된다.
        # x : 좌우, y : 상하, z : 앞뒤
        # 맨 뒤의 1만 제거해주면 된다.
        return points_w.squeeze(-1)
    
    def voxel_pooling(self, geometry, contextVector):        
        # geometry : [B, N, 45, 8, 22, 3] 여기에는 카메라 기준 위치가 들어있고 [좌우, 상하, 앞뒤]
        # contextVector : [B, N, 45, 64, 8, 22] 여기에는 깊이와 특징이 들어있음
        B, N, D_step, C, H_img, W_img = contextVector.shape

        W, H, D = 64, 64, 32 # 격자 설정 [좌우, 앞뒤, 상하]

        # geom : [B * N * 45 * 8 * 22, 3] [좌우, 상하, 앞뒤]
        geom = geometry.reshape(-1, 3) 

        # context : [B, N, 45, 8, 22, 64]
        # context : [B * N * 45 * 8 * 22, 64]
        context = contextVector.permute(0, 1, 2, 4, 5, 3)
        context = context.reshape(-1, C)

        # 이제 geom은 어느곳의 위치만이 들어있고
        # 이제 context는 어느곳의 특징만이 들어있음

        # 각 점이 몇번째 배치인지 알려주는 인덱스
        # torch.arrange(B)를 하면 0 ~ B-1까지의 1차원 배열 생성
        # .view(B, 1, 1, 1, 1)하면 [[[[[0], [1], ... ]]]]이런식으로 생기고
        # .expand하고 reshape(-1)을 하면 각 숫자가 N * D_step * H_img * W_img개 만큼 생김
        batch_idx = torch.arange(B, device=self.device).view(B, 1, 1, 1, 1).expand(B, N, D_step, H_img, W_img)
        batch_idx = batch_idx.reshape(-1) # (0, 0, ..., 1, 1, ...)

        # 필터링 및 좌표변환
        nx = ((geom[:, 0] - (-self.x)) / 0.5).long() # 좌우가 0 ~ 63으로 변환됨 좌우
        ny = ((geom[:, 2] - (-self.y)) / 0.5).long() # 상하가 0 ~ 31으로 변환됨 앞뒤
        nz = ((geom[:, 1] - (-self.z)) / 0.5).long() # 앞뒤가 0 ~ 63으로 변환됨 상하

        # 좌표를 이탈한놈이 있는지 확인
        x_valid = (0 <= nx) & (nx < W) # 좌우
        y_valid = (0 <= ny) & (ny < H) # 앞뒤
        z_valid = (0 <= nz) & (nz < D) # 상하
        valid_mask = x_valid & y_valid & z_valid
        
        # 좌표를 이탈한 놈들은 사라짐
        nx = nx[valid_mask] # 좌우
        ny = ny[valid_mask] # 앞뒤
        nz = nz[valid_mask] # 상하
        
        # 최종적으로 좌표값이 (0~63, 0~31, 0~63)인 데이터들의 모임이 됨
        # 사라지는건 같이 사라지니 괜찮음
        # geom = torch.stack((nx, ny, nz), dim=1) # [_, 3]
        context = context[valid_mask] # [_, 64]
        batch_idx = batch_idx[valid_mask]

        # batch_idx에는 범위 안의 좌표들의 Batch Number가 있음
        # 각 Batch Number에 (W * D * H)를 곱하면 절대로 다른 배치를 넘어서지 못하게 됨
        # 살아남은 놈들에게 부여할 idx를 생성
        idx = batch_idx * (W * H * D) + nx * (H * D) + ny * D + nz
        # [B * W * D * H, C]크기의 0만 있는 2차원 배열 생성
        final_bev = torch.zeros((B * W * H * D, C), device=self.device)
        # final_bev에서 idx의 위치에 context를 더해줌
        # final_bev[idx[0]] += context[0] 와 같음
        final_bev.index_add_(0, idx, context)
        # 이걸 다시 [B, W, H, D, C]로 펴주고
        # [B, C, D, H, W]로 바꿔줌
        final_bev = final_bev.view(B, W, H, D, C).permute(0, 4, 3, 1, 2)

        # [B, C, 32, 64, 64] 상하, 좌우, 앞뒤
        return final_bev