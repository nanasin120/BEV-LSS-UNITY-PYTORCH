# BEV-LSS-UNITY-PYTORCH
본 프로젝트는 [Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D]논문을 바탕으로 했음을 알립니다.

# 1. 프로젝트 소개
본 프로젝트는 Unity가상환경에서 LSS(Lift-Splat-Shoot) 알고리즘을 이용해 주변 환경을 실시간 3D Voxel로 구현하는 프로젝트입니다.

## Tech Stack
* Environment: Unity (가상 시뮬레이션 및 데이터 생성)
* Communication : ML-Agents (Unity-Python 실시간 데이터 송수신)
* Deep Learning : Pytorch (LSS모델 설계 및 학습)
* Visualization : Open3D (실시간 3D Voxel 렌더링)

# 2. 데이터 얻기
## 2.1 가상 환경 조성
Unity로 가상 도시를 만들때 사용한 [도시 에셋](https://assetstore.unity.com/packages/3d/environments/urban/city-package-107224)

Unity내에서 조종하는 [자동차 에셋](https://assetstore.unity.com/packages/3d/vehicles/land/simple-cars-pack-97669)

위의 두 에셋을 이용해 다음과 같이 가상 환경을 만들었습니다.

<img width="1671" height="775" alt="image" src="https://github.com/user-attachments/assets/7f943964-28df-4a23-a1ab-83be5fa622b7" />

## 2.2 이미지 데이터
이미지 데이터는 자동차에 6개의 카메라를 달아서 얻었습니다.
<img width="2027" height="787" alt="image" src="https://github.com/user-attachments/assets/e121c194-ac65-4667-8a3d-8c1a36400dca" />

이미지는 Height 128, Width 352로 저장했습니다. 해상도는 LSS논문에 나와있던 이미지의 해상도와 동일하게 했습니다.

## 2.3 외부 행렬, 내부 행렬
외부 행렬에는 카메라의 회전과 위치가 들어갑니다.

내부 행렬에는 카메라의 초점거리와 주점이 들어갑니다.

해당 데이터들은 스크립트내에서 계산되어 저장됩니다.

## 2.4 데이터 생성 스크립트
데이터를 저장하는 스크립트를 자세히 설명하지는 않겠습니다. 

먼저 데이터를 저장할 위치가 필요합니다.
```
public string datasetRoot = "자신이 저장하고 싶은 위치";
private string currentSavePath, ExtrinsiccsvPath, IntrinsicCsvPath;
```
위치는 본인의 컴퓨터에 맞춰서 하면 됩니다.
```
// 날짜_시간으로 폴더 생성
string timeStmamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
currentSavePath = Path.Combine(datasetRoot, "Log_" + timeStmamp);
```
currentSavePath를 통해 데이터를 저장하는 시점마다 새로운 폴더를 만들어 저장했습니다.
```
// 저장할 폴더 생성
Directory.CreateDirectory(Path.Combine(currentSavePath, "images"));
Directory.CreateDirectory(Path.Combine(currentSavePath, "voxels"));
```
### 이미지
이미지는 Height 128, Width 352로 총 6개의 이미지를 촬영해 저장합니다.

### 외부 행렬
[현재까지 저장한 frame의 개수, 카메라 인덱스, 외부행렬 내용들]이 csv형식으로 저장됩니다.

### 내부 행렬
[초점거리_x, 0, 중심점_x, 0, 초점거리_y, 중심점_y, 0, 0, 1]

위 처럼 9개의 데이터가 저장됩니다.

### 3D Voxel
자동차를 원점으로 해서 복셀의 크기는 0.5, 복셀의 개수는 (64, 32, 64)의 3D Voxel을 생성합니다.

# 3. 모델 구현
모델은 아래와 같은 형태로 구성됩니다.

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
각 형태의 자세한 설명은 아래의 폴더에 적어놓았습니다.

# 4. 학습
## 4.1 하이퍼 파라미터
설정한 하이퍼파라미터들입니다.
| 항목 | 설정값 | 비고 |
| --- | --- | --- |
| Epoch | 55 | 최종 학습 횟수 |
| Batch Size | 8 | 이 이상을 늘릴경우 메모리 부족으로 인해 연산이 느려짐 |
| Learning rate | 0.0001 | 초기 학습률 |
| Weight Decay | 0.01 | 과적합 방지를 위한 가중치 감쇠 |
| Image Size | 128 x 352 | LSS 논문에 의거 |
| num_classes | 4 | 빈공간, 도로, 자동차(자기자신), 장애물 |
| weight | [1.0, 5.0, 2.0, 10.0] | 장애물을 잘 파악하기 위해 가장 큰 가중치 부여 |

## 4.2 데이터셋 구성
데이터셋은 총 4000개의 세트를 이용했습니다.

각 세트에는 이미지 6장, 외부 행렬, 내부 행렬, 정답 3D Voxel이 들어있습니다.

이를 학습과 테스트에 8:2의 비율로 나누었습니다.

## 4.3 손실함수
손실함수로는 CrossEntropy와 DiceLoss를 1 : 9의 비율로 사용했습니다.

DiceLoss의 비율을 9로 한 이유는 학습하며 IOU를 확인한 결과 장애물의 IOU가 계속 정체되어있었기 때문입니다.

그렇기에 CrossEntropy와 DiceLoss둘다 가중치를 [빈공간 : 1, 도로 : 5, 자동차 : 2, 장애물 : 10]으로 주었습니다.

## 4.4 옵티마이저
옵티마이저는 논문에 기반하여 AdamW를 사용하였으며 20에포크마다 학습률을 0.5만큼 떨어뜨리는 스케쥴러 또한 사용했습니다.

# 5. 결과
