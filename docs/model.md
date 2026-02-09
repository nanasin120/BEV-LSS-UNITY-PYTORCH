| 단계 | 이름 | 기능 |
|---|---|---|
| Lift | ImageBackbone(x) | 입력된 이미지 데이터 x를 EfficientNet-B0에 넣어 x_skip과 x_final을 추출한다. |
| | upsampling(x_final, x_skip) | x_final과 x_skip을 합쳐 정보가 더 농축된 데이터 x_up을 뽑는다. |
| | depthnet(x_up) | x_up에서 depth와 feature를 뽑아낸다.|
| | contextVector | depth와 feature를 모두 갖는 contextVector를 만든다. |
| Splat | get_geometry(rots, trans, intrinsics) | 2차원의 좌표를 3차원으로 변환한다. |
| | voxel_pooling(geometry, contextVector) | depth와 feature를 이용해 3차원안에 각 픽셀을 뿌린 voxel_features를 반환한다. |
| Shoot | secondBackbone(voxel_features) | voxel_features의 관계성을 증가시킨 voxel를 반환한다. |
| | taskHead(voxel) | voxel를 실제 복셀로 만든다. |
# Model
다음은 모델의 init에서 설정하는 것들입니다.
```
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
```
각각의 자세한 설명은 실제 사용할때 설명하겠습니다.

LSS는 이미지, 외부행렬의 회전, 외부행렬의 이동, 내부행렬을 입력으로 받습니다.
```
    def forward(self, x, rots, trans, intrinsics):
```
이미지 x는 [B, N, 3, 128, 352]로 [배치 사이즈, 이미지 개수(여기선 6개), RGB, 이미지 높이, 이미지 너비] 입니다.

rots는 [B, N, 3, 3]으로 각 이미지에 대한 3 x 3크기의 행렬입니다.

trans는 [B, N, 3]으로 각 이미지를 찍은 카메라가, 자동차를 원점으로 얼마나 떨어져있는지입니다. x, y, z 좌표라 생각하면 됩니다.

inrinsics는 [B, N, 3, 3]으로 각 이미지에 대한 내부 행렬입니다.

# Lift
## ImageBackbone

ImageBackbone은 이미지를 농축해서 반환합니다.
```
class ImageBackbone(nn.Module):
    def __init__(self):
        super(ImageBackbone, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) # 만들어진 efficientnet_b0 가져옴
        self.backbone = self.model.features[:-1] # 맨 마지막은 제외, 맨 마지막을 320에서 1280으로 만듬
```
이번 프로젝트에서는 논문에 기반하여 EfficientNet_b0를 사용했습니다.

EfficientNet_b0는 굉장히 가볍고 성능도 좋은 Network입니다. 덕분에 저의 노트북에서도 돌릴 수 있었습니다.

EfficientNet_b0의 마지막 레이어는 채널을 320에서 1280으로 확 늘려버리기 때문에 이번 프로젝트에서는 제외했습니다.

```
    def forward(self, x):
        # x : [Batch, N, 3, 128, 352]
        B, N, C, H, W = x.shape
```
이미지를 입력받습니다.

이미지는 위에서 이야기했듯이 [B, N, 3, 128, 352]입니다.

```
        # EfficientNet에 넣기 위해 5차원을 4차원으로 줄여줌
        # [B * N, 3, 128, 352]
        x = x.view(B * N, C, H, W)
```
EfficientNet에 넣기 위해서는 [배치, 채널, 높이, 너비]로 바꿔야 합니다.

그래서 view를 이용해 4차원으로 바꿔줬습니다.
```
        # 채널이 112가 됨
        # [B * N, 112, 8, 22]
        x_skip = x
        for i in range(6): x_skip = self.backbone[i](x_skip)
```
이제 0번 레이어부터 5번 레이어까지 진행시키면 [B * N, 3, 128, 352]였던게 [B * N, 112, 8, 22]로 변하게됩니다.
```
        # 채널이 320가 됨
        # [B * N, 320, 4, 11]
        x_final = x_skip
        for i in range(6, len(self.backbone)): x_final = self.backbone[i](x_final) # 채널이 320가 됨
```
그리고 6번 레이어부터 마지막 레이어까지 돌리게 되면 [B * N, 112, 8, 22]였던게 [B * N, 320, 4, 1]로 변하게 됩니다.

위의 두개 모두 반환할것입니다.

```
        # B * N을 했으니 다시 되돌려줌
        # FC, FH, FW = 112, 8, 22
        FC, FH, FW = x_skip.shape[1], x_skip.shape[2], x_skip.shape[3]
        # [B, N, 112, 8, 22]
        x_skip = x_skip.view(B, N, FC, FH, FW)

        # FC, FH, FW = 320, 4, 11
        # [B, N, 320, 4, 11]
        FC, FH, FW = x_final.shape[1], x_final.shape[2], x_final.shape[3]
        x_final = x_final.view(B, N, FC, FH, FW)
```
모두 view를 이용해 [B * N]이었던걸 [B, N]으로 바꿔줍니다.
```
        # x_final : 정보를 가장 응축해놓은 것
        # x_skip : 그보다 위의, 구체적 위치, 형태 등의 정보가 남아있음
        return x_final, x_skip
```
x_final은 EfficientNet의 끝까지 진행했기 때문에 정보가 많이 응축되어있고 

x_skip은 그에 비하면 덜 농축되었지만 크기는 더 큰 상태입니다.

이제 이 둘을 반환하면 ImageBackbone은 끝나게 됩니다.
```
    def forward(self, x, rots, trans, intrinsics):
        # x_final : (B, N, 320, 4, 11) 이미지 데이터 굉장히 농축됨
        # x_skip : (B, N, 112, 8, 22) 이미지 데이터 덜 농축됨, 그래서 형태가 남아있음
        x_final, x_skip = self.imageBackbone(x)
```
## up
이제 두 데이터를 합쳐서 하나의 데이터로 만들것입니다.
```
        x_final, x_skip = self.imageBackbone(x)

        # x_up : [B, N, 512, 8, 22]
        # x_final과 x_skip을 합쳐 형태가 남아있는 굉장히 농축된 데이터가 됨
        x_up = self.up1(x_final, x_skip)
```
이를 위해 UP이라는 class를 만들어줬습니다.
```
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
```
self.up은 크기를 2배로 만들어주고 conv는 채널을 out_channels로 만듬과 동시에 인접픽셀의 정보를 얻게됩니다.
```
    def forward(self, x1, x2):
        # 연산이 가능하게 맞추어줌
        B1, N1, C1, H1, W1 = x1.shape # [B, N, 320, 4, 11] x_final
        B2, N2, C2, H2, W2 = x2.shape # [B, N, 112, 8, 22] x_skip
        x1 = x1.view(B1 * N1, C1, H1, W1) # [B * N, 320, 4, 11]
        x2 = x2.view(B2 * N2, C2, H2, W2) # [B * N, 112, 8, 22]
```
입력으로는 위에서 얻은 x_final과 x_skip을 받습니다. 이번에도 똑같이 [B * N]으로 만들어줍니다.
```
        # x1은 EfficientNet에 더 깊이 들어갔기 때문에 Width와 Height가 작음
        # 그래서 upSampling해줌
        # x1 : [B * N, 320, 8, 22]
        x1 = self.up(x1)
```
x1은 x_finanl입니다. 크기가 [4, 11]이기 때문에 upSampling으로 크기를 [8, 22]로 올려 x_skip과 똑같게 만들어줍니다.
```
        # 둘을 붙여줌
        # x1 : [B * N, 432, 8, 22]
        x1 = torch.cat([x2, x1], dim=1)
```
그 후 둘을 cat을 통해 붙여줍니다. [B * N, 320 + 112, 8, 22]로 붙게됩니다.
```
        # 연산해줌
        # x_out : [B * N, 512, 8, 22] 여기서 512는 처음 LSS에서 self.up1을 만들떄 설정한 값임
        x_out = self.conv(x1)
```
그 후 conv에 넣어 주변 픽셀의 정보를 얻고 채널을 늘립니다. 채널을 늘리는것은 이미지에서 얻을 수 있는 특징을 늘리는것과 같습니다.
```
        _, C, H, W = x_out.shape
        # [B, N, 512, 8, 22]
        x_out = x_out.view(B1, N1, C, H, W)
        return x_out
```
마지막으로 원래 [B, N]으로 되돌린 후 반환하면 됩니다.
```
        x_final, x_skip = self.imageBackbone(x)

        # x_up : [B, N, 512, 8, 22]
        # x_final과 x_skip을 합쳐 형태가 남아있는 굉장히 농축된 데이터가 됨
        x_up = self.up1(x_final, x_skip)
```
이제 x_final과 x_skip을 합쳐 크기가 [8, 22]이며 특징을 512개 뽑아낸 데이터가 완성됩니다.

## depthnet
```
        x_up = self.up1(x_final, x_skip)

        # depth : [B, N, 45, 8, 22] 깊이에 대한 값이 확률 분포로 나타나있음
        # featrue : [B, N, 64, 8, 22] 특징 64개가 있음
        depth, feature = self.depthnet(x_up)
```
depthnet은 깊이와 특징을 뽑아내는 network입니다.
```
class depthNet(nn.Module):
    def __init__(self, D, C):
        super(depthNet, self).__init__()
        self.D = D # 깊이 45
        self.C = C # 특징 64
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)
```
D는 깊이의 개수, C는 특징의 개수입니다. 처음 LSS의 init에서 depthNet을 만들때 D=45, C=64로 설정했습니다.
```
    def forward(self, x):
        # x는 [B, N, 512, 8, 22]
        B, N, C, H, W = x.shape
        # x : [B * N, 512, 8, 22]
        x = x.view(B * N, C, H, W)
```
입력으로는 위의 up에서 얻은 x_up이 들어옵니다.

다시 conv2d에 넣기 위해 B와 N을 합쳐줍니다.

```
        # x : [B * N, 45 + 64, 8, 22]
        x = self.depthnet(x)
```
그 후 depthnet에 넣어줍니다. 이제 [B * N, 45 + 64, 8, 22]의 데이터를 얻게 됩니다.

여기서 45까지는 깊이로 설정할 것이고 45부터는 특징으로 설정할 것입니다.

```
        depth = x[:, :self.D] # depth : [B * N, 45, 8, 22]
        depth = F.softmax(depth, dim=1) # 깊이를 기준으로 확률분포를 생성, 어느 깊이가 가장 가능성 있는 깊이인지 알 수 있음
```
self.D까지, 즉 0 ~ 44까지를 새로 선언한 depth안에 넣습니다. 이 depth의 의미는 각 m에 무언가 있을 확률입니다.

depth[0]에는 0m에 무언가 있을 확률, depth[1]에는 1m에 무언가 있을 확률입니다.

그렇기 때문에 softmax로 확률분포를 만들어줍니다.

depth[B * N, 30, 2, 9]는 좌표 (2, 9)앞 30m에 무언가 있을 확률 입니다.
```
        feature = x[:, self.D:] # featrue : [B * N, 64, 8, 22]
```
그리고 self.D, 즉 45부터 끝까지는 새로 선언한 feature에 넣어줍니다. feature는 말 그대로 특징입니다.

```
        depth = depth.view(B, N, self.D, H, W) # depth : [B, N, 45, 8, 22]
        feature = feature.view(B, N, self.C, H, W) # featrue : [B, N, 64, 8, 22]

        return depth, feature
```
이 둘을 원래 형태로 돌린뒤 반환하면 depthnet은 끝나게 됩니다.
## contextVector
이제 contextVector를 만들게 됩니다.
```
        depth, feature = self.depthnet(x_up)

        # depth.unsqueeze(3) : [B, N, 45, 1, 8, 22]
        # feature.unsqueeze(2) : [B, N, 1, 64, 8, 22]
        # contextVector : [B, N, 45, 64, 8, 22]
        # 깊이와 특징을 둘 다 가진 contextVector 가완성됨
        contextVector = depth.unsqueeze(3) * feature.unsqueeze(2)
```
contextVector는 깊이와 특징을 합친 데이터입니다.

둘을 합쳐주기 위해 unsqueeze로 빈 차원을 만들어주고 둘을 곱해줍니다.

depth.unsqueeze(3) : [B, N, 45, 1, 8, 22]

feature.unsqueeze(2) : [B, N, 1, 64, 8, 22]

이 둘에서 값이 같은 차원은 그대로 넣어지고 1인 부분은 1이 아닌 값에 들어가게 된다 생각하면 됩니다.

그렇게 contextVector는 [B, N, 45, 64, 8, 22]의 차원을 갖게 됩니다.

contextVector는 [B, N, 45, 64, 8, 22]의 형태를 가졌으며 

예를 들어 [B, N, 20, 30, 2, 10]의 경우 좌표(2, 10)앞 20m의 30번째 특징의 확률적 강도입니다.

30번째 특징이 뭔지는 모르지만 20m앞에 이게 있을 확률이 대충 [B, N, 20, 30, 2, 10]정도 인겁니다.

---

지금까지 한건 2차원 이미지를 깊이를 갖는 3차원으로 끌어올리는 작업입니다.

# Splat
## get_geometry
```
        contextVector = depth.unsqueeze(3) * feature.unsqueeze(2)

        # geometry : [B, N, 45, 8, 22, 3]
        # 0 : 좌우, 1 : 상하, 2 : 앞뒤
        geometry = self.get_geometry(rots, trans, intrinsics)
```
get_geometry는 위에서 만든 contextVector를 자동차를 중심으로 하는 3차원 공간에 매핑하기위한 공간을 만듭니다.

get_geometry를 하기 전 먼저 frustum을 만들어야 합니다.

### create_frustum

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/4a5eb746-175d-4b9a-ad5b-06c8689a5d98" />

제미나이를 통해 그림을 만들었습니다.

그림에서 중요한 부분은 하얀 점들이 찍힌 3차원 공간입니다.

코드를 통해 저 3차원 공간을 만들것입니다.
```
    def create_frustum(self):
        # ogfH, ogfW는 격자 블록의 크기 [8, 22]
        # Output Grid Final Height/Width
        # 지금까지 만든 [8, 22]의 평면을 3차원으로 Shoot할 발사대를 만들거임
        ogfH, ogfW = self.final_H, self.final_W

        full_W, full_H = 352, 128
```
Output Grid Final Height/Width는 격자 블록의 크기입니다. 그림 속 격자 하나하나가 8x22의 크기를 갖습니다.

self.final_H와 self.final_W도 처음 LSS의 init를 설정할때 8, 22로 설정해놓은 값입니다.

full_W와 full_H는 실제 이미지의 크기와 갖습니다. 이유는 뒤에 나오게됩니다.

```
        # 0 ~ ogfW-1까지 ogfW개의 1차원 행렬 생성
        xs = torch.linspace(0, full_W - 1, ogfW).float().to(self.device)

        # 0 ~ ogfH-1까지 ogfH개의 1차원 행렬 생성
        ys = torch.linspace(0, full_H - 1, ogfH).float().to(self.device) 
```
xs는 0부터 full_W-1까지 ogfW개의 1차원 행렬입니다. 이는 너비에 이용될것입니다.

ys는 0부터 full_H-1까지 ogfH개의 1차원 행렬입니다. 이는 높이에 이용될것입니다.

```
        # self.D_min부터 self.D_max까지 self.D_step만큼 증가하는 1차원 행렬 생성
        # 1, 46, 1이니 ds = [1, 2, 3 ... , 43, 44, 45]
        ds = torch.arange(self.D_min, self.D_max, self.D_step).float().to(self.device)
```
ds는 깊이입니다. self.D_min은 제가 설정한 깊이의 최솟값입니다. self.D_max는 깊이의 최댓값이고 self.D_step은 깊이의 증가값입니다.

1, 46, 1로 설정했기 때문에 1부터 45까지 1씩 증가하는, 즉 [1, 2, 3, ... , 44, 45]의 1차원 행렬이 됩니다.

