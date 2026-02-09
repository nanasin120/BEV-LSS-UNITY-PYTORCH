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
<img width="537" height="474" alt="image" src="https://github.com/user-attachments/assets/0a9d7215-7354-4b7a-a9f4-d6e568543a08" />

frustum은 간단하게 부채꼴이라 생각하면 됩니다.

지금 만들 frustum은 밑면과 윗면이 사각형인 3차원의 부채꼴입니다.

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


처음 입력으로 주어지는 이미지를 생각해봅시다.
<img width="1118" height="457" alt="image" src="https://github.com/user-attachments/assets/0c74d14d-edc6-4144-8f67-f6fb426b9bfd" />
너비가 352이고 높이가 128입니다.

이 이미지를 ofg, 즉 8개와 22개로 잘라보겠습니다.

<img width="1169" height="473" alt="image" src="https://github.com/user-attachments/assets/01f69a14-0b00-48e7-927a-ea1bd13b45e1" />

다음과 같이 16x16의 블럭이 나옵니다. 저희가 구한 xs와 ys는 모두 저 블럭을 의미하는 것입니다.

<img width="1149" height="455" alt="image" src="https://github.com/user-attachments/assets/024c4827-3232-49ed-8702-66d4afe51ef1" />

xs와 ys를 인덱스로 생각하면 됩니다. 이미지를 8,22개로 자른것입니다.

이것들을 하나로 만들어줄것입니다.
```
        # (ds, ys, ws)인 3차원 행렬 생성, d에는 ds만 적혀있고 y에는 ys만 적혀있고 x에는 xs만 적혀있다.
        # [45, 8, 22]인 3차원 행렬이 생성되는것
        # [깊이, 높이, 너비]인 상태로
        # d에는 깊이, y에는 높이, x에는 너비가 중심으로 들어가게됨
        d, y, x = torch.meshgrid(ds, ys, xs, indexing='ij') 
```
torch.meshgrid를 통해 크기가 [45, 8, 22]인 3차원 행렬을 만들어줍니다. 이때 반환되는 값은 총 3개 입니다.

처음 d에는 오직 ds의 값만이 들어있습니다. [30, 2, 10]의 값은 30이고 [30, 1, 2]의 값도 30입니다.

y에는 당연히 ys의 값만들어있습니다. [30, 2, 10]의 값은 2이고 [30, 1, 2]의 값은 1입니다.

x에는 당연히 xs의 값만들어있습니다. [30, 2, 10]의 값은 10이고 [30, 1, 2]의 값은 2입니다.

```
        # 맨 뒤에 3개를 연속으로 쌓아 [45, 8, 22, 3]의 4차원 행렬 생성
        # 이제 [45, 8, 22]를 보면 3개의 x, y, d값을 볼 수 있음
        # x와 y에 d를 곱하는 이유는 깊이에 따라 좌표도 함께 멀어지기 때문이다.
        frustum = torch.stack([x * d, y * d, d], dim=-1)
```
이들을 하나로 쌓아 4차원 행렬을 만들어주는데 이때 x와 y에 d를 곱해줍니다. 각 좌표에 깊이를 추가해 크기를 늘려주는 것입니다.

깊이라고는 하지만 사실상 거리입니다. 간단하게 생각해서, 가까이 있는 물체는 크게 보입니다. 하지만 멀리 있는 물체는 작게 보입니다.

멀리있는 픽셀을 보고있다 생각하면, 거리가 먼 픽셀은 작게 보입니다. 가까이서 보면 지금 보는것보다 크게 보이는것입니다.

그러니 d를 곱해 픽셀의 크기를 크게해주는 것입니다. 

<img width="843" height="665" alt="image" src="https://github.com/user-attachments/assets/ef3b3179-4dbb-4d17-ab27-b0351c1b7be2" />

그러면 위의 그림처럼 거리가 먼 픽셀들의 크기는 거리가 가까운 픽셀들에 비해 크기가 커지게 됩니다.

```
        frustum = torch.stack([x * d, y * d, d], dim=-1)

        return frustum
```
이렇게 만든 frustum을 반환하면 create_frustum은 끝입니다.

지금까지 만든 frustum은 완벽한 3차원 좌표계가 아닙니다 완벽한 3차원 좌표계를 만들기 위한 준비물입니다.

다시 get_geometry로 돌아가겠습니다.

```
    def get_geometry(self, rots, trans, intrinsics):
        B, N = intrinsics.shape[0], intrinsics.shape[1] # 배치와 이미지개수
        frustum = self.frustum # [45, 8, 22, 3] = [깊이, 높이, 너비, xyz]

        inv_intrinsics = torch.inverse(intrinsics) # 내부 행렬 뒤집기 inv_intrinsics : [B, N, 3, 3]
```
내부 행렬로 부터 배치와 이미지의 개수를 얻습니다. 그리고 frustum을 만든 후 내부 행렬을 뒤집습니다.

```
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
```
이제 필홀 카메라 공식을 통해 frustum을 실제 3차원 좌표로 만들어줍니다.
<img width="791" height="587" alt="image" src="https://github.com/user-attachments/assets/53b22d28-32a3-447f-9b67-43a71b09e1e0" />

위에서 f는 카메라 렌즈에서 이미지 센서까지의 거리, d는 렌즈에서 실제 물체까지의 거리이고 A는 실제 물체의 크기, a는 카메라에 찍힌 크기입니다.

이때 a/f = A/d의 공식이 성립하고 A = d * (a/f)의 공식이 성립하게 됩니다.

행렬에서 나누기를 하려면 역행렬을 구한뒤 곱하면 됩니다. 그걸 하는것입니다.

d를 곱하지 않는 이유는 d는 0~44까지 전부다 구하기 때문입니다. 그리고 애시당초 frustum을 구할때 이미 d는 넣어놨습니다.

이제 points_c [B, N, 45, 8, 22, 3, 1]은 거리는 45m 떨어졌고 좌표는 (8, 22)인 픽셀이 

카메라 중심에서 좌우로 얼마나 떨어졌나 (x), 상하로 얼마나 떨어졌나 (y), 정면으로 부터 얼마나 멀리있나 (d)입니다.

```
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
```
지금의 points_c안 모든 픽셀은 자동차를 기준으로 정면에만 있습니다. 하지만 실제 카메라는 자동차를 기준으로 원처럼 둘러있습니다.

이는 README.md에 있는 이미지에 나와있습니다.

이제 points_c를 강체변환 공식을 통해 회전시켜주는것입니다. 3차원으로 끌어올린 픽셀들은 자동차를 기준으로 하는 3차원으로 뿌려주는것입니다.

강체변환에 대한 증명은 일단 넘어가겠습니다.

```
        # 이제 [B, N, 45, 8, 22, 3, 1]를 [B, N, 45, 8, 22, 3]로 만들어준다.
        # N번 카메라의 (깊이, 높이, 너비)는 자동차 정면을 기준으로 회전한 x, y, z를 갖게 된다.
        # x : 좌우, y : 상하, z : 앞뒤
        # 맨 뒤의 1만 제거해주면 된다.
        return points_w.squeeze(-1)
```
마지막의 의미없는 1만 제거해주고 반환하면 됩니다.

points_w는 자동차를 원점으로 하는 3차원 좌표계에서 각 픽셀의 좌표입니다.

## Voxel_pooling
```
        geometry = self.get_geometry(rots, trans, intrinsics)

        # 미완성 복셀맵이 나옴 [B, 64, 32, 64, 64] 상하, 좌우, 앞뒤
        voxel_features = self.voxel_pooling(geometry, contextVector)
```
얻은 geometry와 contextVector를 합치면 이제 자동차를 원점으로 하는 3차원 좌표계에서 각 픽셀의 의미를 알 수 있습니다.
```
    def voxel_pooling(self, geometry, contextVector):        
        # geometry : [B, N, 45, 8, 22, 3] 여기에는 카메라 기준 위치가 들어있고 [좌우, 상하, 앞뒤]
        # contextVector : [B, N, 45, 64, 8, 22] 여기에는 깊이와 특징이 들어있음
        B, N, D_step, C, H_img, W_img = contextVector.shape
```
먼저 contextVector의 shape를 저장해줍니다.
```
        W, H, D = 64, 64, 32 # 격자 설정 [좌우, 앞뒤, 상하]
```
저는 격자, 즉 복셀의 크기를 위와 같이 설정했습니다. 좌우로 64개, 앞뒤로 64개, 상하로 32개입니다.
```
        # geom : [B * N * 45 * 8 * 22, 3] [좌우, 상하, 앞뒤]
        geom = geometry.reshape(-1, 3)
```
geom은 geometry를 좌표, 즉 x, y, z만 남기고 전부 합쳐버린 것입니다.

이때 x가 좌우, y가 상하, z(d)가 앞뒤입니다. d거리에 있는 이미지를 생각하면 연상이 잘됩니다.

```
        # context : [B, N, 45, 8, 22, 64]
        # context : [B * N * 45 * 8 * 22, 64]
        context = contextVector.permute(0, 1, 2, 4, 5, 3)
        context = context.reshape(-1, C)
```
contextVector는 특징만 남기고 전부 합쳐줍니다. 여기서 주의할점은 합치는 순서를 같게 하는 것입니다.

geometry와 contextVector 둘 다 N * N * 45 * 8 * 22의 순서로 합친걸 알 수 있습니다.

이제 geom은 위치, context는 특징만이 들어있습니다. 

```
        # 각 점이 몇번째 배치인지 알려주는 인덱스
        # torch.arrange(B)를 하면 0 ~ B-1까지의 1차원 배열 생성
        # .view(B, 1, 1, 1, 1)하면 [[[[[0], [1], ... ]]]]이런식으로 생기고
        # .expand하고 reshape(-1)을 하면 각 숫자가 N * D_step * H_img * W_img개 만큼 생김
        batch_idx = torch.arange(B, device=self.device).view(B, 1, 1, 1, 1).expand(B, N, D_step, H_img, W_img)
        batch_idx = batch_idx.reshape(-1) # (0, 0, ..., 1, 1, ...)
```
batch_idx는 데이터가 어느 배치에서 나온 데이터인지를 알려주는 index입니다.

arrange로 0부터 B-1까지 숫자를 만든뒤 expand를 통해 각 숫자를 N * D_step * H_img * W_img개 만큼 추가하고 reshape로 한줄로 쭉 펴줍니다.

```
        # 필터링 및 좌표변환
        nx = ((geom[:, 0] - (-self.x)) / 0.5).long() # 좌우가 0 ~ 63으로 변환됨 좌우
        ny = ((geom[:, 2] - (-self.y)) / 0.5).long() # 상하가 0 ~ 31으로 변환됨 앞뒤
        nz = ((geom[:, 1] - (-self.z)) / 0.5).long() # 앞뒤가 0 ~ 63으로 변환됨 상하
```
이제 필터링과 좌표변환을 할것입니다. 

geom[:, 0]에는 모든 x가 모여있습니다. 여기에 LSS의 init에서 설정한 self.x (16)을 더하고 여기에 2를 곱한걸 정수로 바꿉니다.

저의 목표는 자동차를 원점으로 -32 ~ 32안에 있는 좌표만 살리는 것입니다.

예를 들어 원래 좌표가 -10이었다 하고 ((-10 - (-16)) / 0.5) = 12입니다. 이 좌표는 살아남게됩니다.

그냥 self.x를 더하는것에서 끝나지 않고 0.5를 곱해주는 이유는 복셀의 크기를 0.5로 설정했기 때문입니다.

```
        # 좌표를 이탈한놈이 있는지 확인
        x_valid = (0 <= nx) & (nx < W) # 좌우
        y_valid = (0 <= ny) & (ny < H) # 앞뒤
        z_valid = (0 <= nz) & (nz < D) # 상하
        valid_mask = x_valid & y_valid & z_valid
```
이제 위에서 설정한 W, H, D안에 있는 좌표만 true로 만들고 나머지는 false로 만든 mask를 전부 합쳐줍니다.

```
        # 좌표를 이탈한 놈들은 사라짐
        nx = nx[valid_mask] # 좌우
        ny = ny[valid_mask] # 앞뒤
        nz = nz[valid_mask] # 상하

        # 최종적으로 좌표값이 (0~63, 0~31, 0~63)인 데이터들의 모임이 됨
        # 사라지는건 같이 사라지니 괜찮음
        context = context[valid_mask] # [_, 64]
        batch_idx = batch_idx[valid_mask]
```
그 후 mask를 적용시켜주면 목표로 하는 0~63 안에 있는 좌표만 남고 나머지는 사라지게 됩니다.

mask는 context와 batch_idx에도 적용시켜줍니다.
```
        # batch_idx에는 범위 안의 좌표들의 Batch Number가 있음
        # 각 Batch Number에 (W * D * H)를 곱하면 절대로 다른 배치를 넘어서지 못하게 됨
        # 살아남은 놈들에게 부여할 idx를 생성
        idx = batch_idx * (W * H * D) + nx * (H * D) + ny * D + nz
```
batch_idx에는 좌표 batch number가 있습니다. 모든 좌표는 (0, 0, 0)에서 시작해서 (63, 63, 31)까지 가게됩니다.

즉 0번 배치안에 있는 모든 좌표의 개수를 더해도 64 * 64 * 32보단 작게됩니다. 각 배치에 이를 곱해줘 다른 배치를 침범하지 못하게 합니다.

그리고 x, y, d의 순서로 batch_idx에 더해줍니다. 이때 각각 필요한 만큼 곱해줘야합니다.

지금 이건 3차원의 배열을 1차원의 배열로 쭉 피는 과정과 같습니다. 그렇기 때문에 자신보다 낮은 차원의 크기만큼 곱해주는 것입니다.
```
        # [B * W * H * D, C]크기의 0만 있는 2차원 배열 생성
        final_bev = torch.zeros((B * W * H * D, C), device=self.device)
```
final_bev를 만들어줍니다. [B * W * H * D, C]의 0만 있는 2차원 배열입니다. 여기서 C는 contextVector에서 가져온 것입니다.
```
        # final_bev에서 idx의 위치에 context를 더해줌
        # final_bev[idx[0]] += context[0] 와 같음
        final_bev.index_add_(0, idx, context)
```
index_add_를 통해 final_bev에 context에 있는 특징들을 각자의 자리에 맞게 전부 더해줍니다. 

idx는 위에서 만든 1차원 배열로 3차원 좌표의 인덱스를 1차원으로 쭉 늘린것입니다.

context에 있는 [B, N, 45, 8, 22]의 특징 [C(64)]를 각자의 자리에 전부 더해주는 것입니다.

```
        # 이걸 다시 [B, W, H, D, C]로 펴주고
        # [B, C, D, H, W]로 바꿔줌
        final_bev = final_bev.view(B, W, H, D, C).permute(0, 4, 3, 1, 2)

        # [B, C, 32, 64, 64] 상하, 좌우, 앞뒤
        return final_bev
```
이걸 펴준뒤 바꿔주면 완성입니다. 이때 펴주는 순서를 주의해야합니다.

더하는 순서가 [B, W, H, D, C]였으니 [B, W, H, D, C]로 펴줘야합니다.

이걸 다시 바꾸면 final_bev는 [배치, 상하, 좌우, 앞뒤 특징]으로 이루어지게 됩니다.

# Shoot
## secondBackbone
```
        voxel_features = self.voxel_pooling(geometry, contextVector)

        # [B, 64, 32, 64, 64] 더 관계가 강화되어서 나옴 [상하, 좌우, 앞뒤]
        voxel = self.secondBackbone(voxel_features)
```
만들어진 voxel_features를 secondBackbone에 넣어줍니다.

```
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
```
secondBackbone은 3차원 conv로 옆에있는 픽셀과의 관계성을 더욱 높여줍니다.

## TaskHead
```
        # [B, 64, 32, 64, 64] 더 관계가 강화되어서 나옴 [상하, 좌우, 앞뒤]
        voxel = self.secondBackbone(voxel_features)

        # [B, 4, 32, 64, 64] 상하, 좌우, 앞뒤
        head = self.taskHead(voxel)
```
이제 마지막으로 실제 복셀을 만듭니다.
```
class TaskHead(nn.Module):
    def __init__(self, in_channel=64, out_channel=4):
        super(TaskHead, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        
    def forward(self, x):
        out = self.conv(x)

        return out
```
TaskHead는 채널을 64개에서 4개로 줄여버립니다. 이는 복셀에서 표현하는 [빈공간, 도로, 자동차, 장애물]입니다.

# 마무리
```
        # [B, 4, 32, 64, 64] 상하, 좌우, 앞뒤
        head = self.taskHead(voxel)

        return head
```
이로서 기나긴 설명이 끝이났습니다.

슬프게도 처음 ImageBackbone에서 이미지의 크기를 8, 22로 줄여버리기때문에 작은 픽셀, 예를 들어 나무, 전봇대 같은 장애물은 사라져버리게됩니다.

입력 이미지의 크기를 늘리거나 다른 network를 이용하는것도 좋을 것 같습니다.
