using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class DataAgent : Agent
{
    public Camera[] Cameras;
    private int Width = 352, Height = 128;
    public Vector3 gridSize = new Vector3(64, 32, 64); // 가로, 높이, 세로 [복셀의 개수]
    public float voxelSize = 0.25f; // 복셀 한 칸 크기
    
    public override void CollectObservations(VectorSensor sensor)
    {
        Matrix4x4 egoToWorld = transform.localToWorldMatrix;
        Matrix4x4 worldToEgo = egoToWorld.inverse;

        foreach (Camera cam in Cameras)
        {
            // 외부 행렬
            Matrix4x4 camToWorld = cam.transform.localToWorldMatrix;
            Matrix4x4 camToEgo = worldToEgo * camToWorld;

            Matrix4x4 convert = Matrix4x4.identity;
            convert[1, 1] = -1.0f;
            Matrix4x4 finalMat = camToEgo * convert;

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    sensor.AddObservation(finalMat[i, j]);
            
            // 내부 행렬
            float f_y = Height / (2.0f * Mathf.Tan(0.5f * cam.fieldOfView * Mathf.Deg2Rad));
            float f_x = f_y;

            float c_x = Width / 2.0f;
            float c_y = Height / 2.0f;

            sensor.AddObservation(f_x);
            sensor.AddObservation(f_y);
            sensor.AddObservation(c_x);
            sensor.AddObservation(c_y);
            
            // [ fx  0  cx ]
            // [  0 fy  cy ]
            // [  0  0   1 ]
            
        }

        // Vector3 origin = transform.position
        //     - (transform.right * (gridSize.x * voxelSize) / 2)
        //     - (transform.up * (gridSize.y * voxelSize) / 2)
        //     - (transform.forward * (gridSize.z * voxelSize) / 2); // 자동차 위치를 가운데로 두고 가장 끝쪽으로 이동

        // for(int x = 0; x < gridSize.x; x++)
        // {
        //     for(int y = 0; y < gridSize.y; y++)
        //     {
        //         for(int z = 0; z < gridSize.z; z++)
        //         {
        //             Vector3 worldPos = origin 
        //                 + (transform.right * (x * voxelSize + voxelSize/2))
        //                 + (transform.up * (y * voxelSize + voxelSize/2))
        //                 + (transform.forward * (z * voxelSize + voxelSize/2)); // 끝쪽에서 시작해서 조금씩 이동하는거임

        //             Collider[] hits = Physics.OverlapBox(worldPos, Vector3.one * (voxelSize / 2.1f), transform.rotation); // 해당 위치에 부딫히는게 있는지 확인

        //             byte val = 0; // 빈 공간
        //             int maxPriority = 0;

        //             if (hits.Length > 0) // 부딫힌거 있으면 확인
        //             {
        //                 foreach (var hit in hits) // 잡힌 모든 물체를 다 검사함
        //                 {
        //                     string t = hit.tag;
        //                     int currentPriority = 0;
        //                     byte currentVal = 0;

        //                     if (t == "Obstacle" || t == "Building" || t == "sign") 
        //                     { 
        //                         currentVal = 3; 
        //                         currentPriority = 3; 
        //                     }
        //                     else if (t == "Car") // 내 차도 여기 포함됨!
        //                     { 
        //                         currentVal = 2; 
        //                         currentPriority = 2; 
        //                     }
        //                     else if (t == "Road") 
        //                     { 
        //                         currentVal = 1; 
        //                         currentPriority = 1; 
        //                     }

        //                     // 더 센 놈이 나타나면 덮어쓰기
        //                     if (currentPriority > maxPriority)
        //                     {
        //                         maxPriority = currentPriority;
        //                         val = currentVal;
        //                     }
        //                 }
        //             }
        //             sensor.AddObservation(val);
        //         }
        //     }
        // }
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        
    }
    public override void OnEpisodeBegin()
    {
        
    }
}
