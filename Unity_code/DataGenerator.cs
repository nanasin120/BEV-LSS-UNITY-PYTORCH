using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;

public class DataGenerator : MonoBehaviour
{
    public string datasetRoot = "C:/Users/MSI/Desktop/DrivingData"; // 데이터셋 저장할 폴더

    public float captureDistance = 1.0f; // 1미터 움직일때마다 저장
    public Camera[] cameras; // 6개 카메라 연결

    // 자동차의 사이즈가 대충 앞뒤로 4, 양옆으로 2, 위아래로 2정도
    public Vector3 gridSize = new Vector3(64, 32, 64); // 가로, 높이, 세로 [복셀의 개수]
    public float voxelSize = 0.5f; // 복셀 한 칸 크기
    public int firstframe = 0;

    private string currentSavePath, ExtrinsiccsvPath, IntrinsicCsvPath;
    private int frameCount = 0;
    private Vector3 lastCapturePos;

    private RenderTexture rt; // 가상의 스크린 만들기 (가로, 세로, 깊이)
    private Texture2D shot; // RenderTexture는 GPU, 이건 cpu, 파일로 저장하려면 cpu로 가져와야함
    private int Width = 352, Height = 128;

    void Start()
    {
        // 날짜_시간으로 폴더 생성
        string timeStmamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        currentSavePath = Path.Combine(datasetRoot, "Log_" + timeStmamp);

        // 저장할 폴더 생성
        Directory.CreateDirectory(Path.Combine(currentSavePath, "images"));
        Directory.CreateDirectory(Path.Combine(currentSavePath, "voxels"));

        lastCapturePos = transform.position;
        Debug.Log($"데이터 수집 시작 : {currentSavePath}");

        // 이미지 저장에 사용하는 것들
        rt = new RenderTexture(Width, Height, 24);
        shot = new Texture2D(Width, Height, TextureFormat.RGB24, false);

        // 외부 행렬 저장용 csv
        ExtrinsiccsvPath = Path.Combine(currentSavePath, "extrinsics.csv"); 
        if (!File.Exists(ExtrinsiccsvPath))
        {
            string header = "frame_id,camera_id,r00,r01,r02,tx,r10,r11,r12,ty,r20,r21,r22,tz,r30,r31,r32,tw\n";
            File.WriteAllText(ExtrinsiccsvPath, header);
        }

        // 내부 행렬 저장용 csv
        IntrinsicCsvPath = Path.Combine(currentSavePath, "intrinsics.csv"); 
        if (!File.Exists(IntrinsicCsvPath))
        {
            // 3x3 행렬이라 데이터가 9개입니다.
            string header = "frame_id,camera_id,fx,0,cx,0,fy,cy,0,0,1\n";
            File.WriteAllText(IntrinsicCsvPath, header);
        }
    }

    void Update()
    {
        float dist = Vector3.Distance(transform.position, lastCapturePos);

        // captureDistance보다 더 움직였으면 저장 시작
        // 현재는 captureDistance를 1로 설정해서 1m 움직이면 시작
        if(dist >= captureDistance)
        {
            StartCoroutine(CaptureFrame());
            lastCapturePos = transform.position;
        }
    }

    IEnumerator CaptureFrame()
    {
        yield return new WaitForEndOfFrame(); // 프레임이 종료될때까지 대기
        
        // 모든 카메라 이미지 저장
        for (int i = 0; i < cameras.Length; i++) {
            SaveImage(cameras[i], i);
            SaveExtrinsic(cameras[i], i);
            SaveIntrinsic(cameras[i], i);
        }
        // 복셀 저장
        SaveVoxel();

        // 지금까지 몇개 찍었나 확인
        frameCount++;
        if(frameCount % 10 == 0) Debug.Log($"데이터 {frameCount}장 수집됨");
    }

    void SaveImage(Camera cam, int idx)
    {
        var prevTarget = cam.targetTexture;

        cam.targetTexture = rt; // 카메라랑 연결, 이러면 게임 화면이 아닌 rt에 보는 화면을 그림
        cam.Render(); // 카메라 셔터 누르는 것
        RenderTexture.active = rt; // 앞으로 RenderTexture로 하는 모든 작업은 rt에 하는거임
        
        shot.ReadPixels(new Rect(0, 0, Width, Height), 0, 0); // rt에 있는거 읽어옴, active해놔서 되는거임 0,0부터 640,480까지
        shot.Apply(); // 복사는 끝냈고 이제 shot안에 저장하는거임

        // shot.EncodeToJPG() : 이미지 데이터를 JPG로 압축 변환
        // File.WriteAllBytes(A, B) : A에 B를 저장함
        File.WriteAllBytes(Path.Combine(currentSavePath, "images", $"frame_{frameCount:D6}_cam_{idx}.jpg"), shot.EncodeToJPG());
        
        // 이제 뒷정리 하는거임
        // 특히 Destory안하면 계속 생성되어서 메모리 누수됨
        cam.targetTexture = prevTarget; // 카메라 원래대로 게임화면에 나오게 돌리고
        RenderTexture.active = null; // 작업 대상 해제하고
    }
    void SaveVoxel()
    {
        int total = (int)(gridSize.x * gridSize.y * gridSize.z); // 전체 복셀 개수
        byte[] data = new byte[total]; // 0은 비어있음, 1은 도로, 2는 자동차, 3은 빌딩, 장애물 등
        int idx = 0; // 현재까지 만든 복셀

        Vector3 origin = transform.position
            - (transform.right * (gridSize.x * voxelSize) / 2)
            - (transform.up * (gridSize.y * voxelSize) / 2)
            - (transform.forward * (gridSize.z * voxelSize) / 2); // 자동차 위치를 가운데로 두고 가장 끝쪽으로 이동

        for(int x = 0; x < gridSize.x; x++)
        {
            for(int y = 0; y < gridSize.y; y++)
            {
                for(int z = 0; z < gridSize.z; z++)
                {
                    Vector3 worldPos = origin 
                        + (transform.right * (x * voxelSize + voxelSize/2))
                        + (transform.up * (y * voxelSize + voxelSize/2))
                        + (transform.forward * (z * voxelSize + voxelSize/2)); // 끝쪽에서 시작해서 조금씩 이동하는거임

                    Collider[] hits = Physics.OverlapBox(worldPos, Vector3.one * (voxelSize / 2.1f), transform.rotation); // 해당 위치에 부딫히는게 있는지 확인

                    byte val = 0; // 빈 공간
                    int maxPriority = 0;

                    if (hits.Length > 0) // 부딫힌거 있으면 확인
                    {
                        foreach (var hit in hits) // 잡힌 모든 물체를 다 검사함
                        {
                            string t = hit.tag;
                            int currentPriority = 0;
                            byte currentVal = 0;

                            if (t == "Obstacle" || t == "Building" || t == "sign") 
                            { 
                                currentVal = 3; 
                                currentPriority = 3; 
                            }
                            else if (t == "Car")
                            { 
                                currentVal = 2; 
                                currentPriority = 2; 
                            }
                            else if (t == "Road") 
                            { 
                                currentVal = 1; 
                                currentPriority = 1; 
                            }

                            // 더 센 놈이 나타나면 덮어쓰기
                            if (currentPriority > maxPriority)
                            {
                                maxPriority = currentPriority;
                                val = currentVal;
                            }
                        }
                    }
                    data[idx++] = val;
                }
            }
        }
        // 저장
        File.WriteAllBytes(Path.Combine(currentSavePath, "voxels", $"frame_{frameCount:D6}_voxel.bin"), data);
    }
    void SaveExtrinsic(Camera cam, int camIdx)
    {
        // Ego 역행렬 * 카메라 행렬 = (World -> Ego) * (Cam->World) = Cam->Ego
        Matrix4x4 egoToWorld = transform.localToWorldMatrix;
        Matrix4x4 camToWorld = cam.transform.localToWorldMatrix;
        Matrix4x4 camToEgo = egoToWorld.inverse * camToWorld;

        Matrix4x4 convert = Matrix4x4.identity;
        convert[1, 1] = -1.0f;

        Matrix4x4 finalMat = camToEgo * convert;

        // 외부 행렬 넣기
        StringBuilder sb = new StringBuilder();
        sb.Append(frameCount.ToString("D6")).Append(",");
        sb.Append(camIdx).Append(",");
        sb.Append(finalMat[0, 0]).Append(",").Append(finalMat[0, 1]).Append(",").Append(finalMat[0, 2]).Append(",").Append(finalMat[0, 3]).Append(",");
        sb.Append(finalMat[1, 0]).Append(",").Append(finalMat[1, 1]).Append(",").Append(finalMat[1, 2]).Append(",").Append(finalMat[1, 3]).Append(",");
        sb.Append(finalMat[2, 0]).Append(",").Append(finalMat[2, 1]).Append(",").Append(finalMat[2, 2]).Append(",").Append(finalMat[2, 3]).Append(",");
        sb.Append(finalMat[3, 0]).Append(",").Append(finalMat[3, 1]).Append(",").Append(finalMat[3, 2]).Append(",").Append(finalMat[3, 3]);
        sb.Append("\n");

        File.AppendAllText(ExtrinsiccsvPath, sb.ToString());
    }
    void SaveIntrinsic(Camera cam, int camIdx)
    {
        float f_y = Height / (2.0f * Mathf.Tan(0.5f * cam.fieldOfView * Mathf.Deg2Rad));
        float f_x = f_y;

        // 중심점(c) 계산 (화면 정중앙)
        float c_x = Width / 2.0f;
        float c_y = Height / 2.0f;

        // 3x3 행렬
        // [ fx  0  cx ]
        // [  0 fy  cy ]
        // [  0  0   1 ]
        
        StringBuilder sb = new StringBuilder();

        // 내부 행렬 넣기
        sb.Append(frameCount.ToString("D6")).Append(","); 
        sb.Append(camIdx).Append(",");                    
        sb.Append(f_x).Append(",").Append(0).Append(",").Append(c_x).Append(",");
        sb.Append(0).Append(",").Append(f_y).Append(",").Append(c_y).Append(",");
        sb.Append(0).Append(",").Append(0).Append(",").Append(1);
        sb.Append("\n");

        File.AppendAllText(IntrinsicCsvPath, sb.ToString());
    }
}
