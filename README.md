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


# 4. 학습

# 5. 결과
