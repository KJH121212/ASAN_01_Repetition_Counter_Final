# [Project] 실시간 재활 운동 측정 및 피드백 시스템

## 1. 프로젝트 개요
별도의 센서나 장비 없이 카메라만으로 재활 운동을 실시간 측정하고 피드백을 제공하는 게임용 프로그램 개발을 최종 목표로 합니다. 병원과 가정에서 추가 장비 없이 운동 상태 분석 및 반복 횟수 자동 계산이 가능하도록 구현하며, 중간 단계로 기울어진 침대에 누워 있는 환자들을 대상으로 한 임상 데이터를 통해 시스템의 정확도와 안정성을 검증하는 것을 목표로 한다.

---

## 2. Data

### 2.1. 데이터 구성 (Data Settings)
본 프로젝트는 기준 모델 학습 및 임상 검증을 위해 다음과 같은 데이터셋을 활용합니다.

| 폴더명 | 데이터 내용 및 규모 | 주요 활용 목적 |
| :--- | :--- | :--- |
| **sample_data** | 총 75개의 주요 재활 운동 샘플 | 동작별 기준 자세 측정 및 모델 학습용 표준 데이터 |
| **AI_dataset** | 일반인 10명의 영상 (3개 각도, 2개 장소) | 29가지 재활 동작 무작위 수행을 통한 모델 범용성 테스트 |
| **Won_Kim_research_at_Bosanjin** | 중환자 15명의 실제 임상 영상 (방문일별 수행) | 실제 임상 환경에서의 2~3가지 재활 동작 분석 및 검증 |
| **Nintendo_Therapy** | 중환자 22명의 실제 임상 영상 (방문일별 수행) | 장기 추적(Visit) 데이터 기반의 재활 경과 및 정확도 분석 |

---

### 2.2. 데이터 관리 구조 (Data Structure)

![Data 폴더 구조](./img/data.png)

- **`1_FRAME/`**: 원본 동영상에서 추출한 개별 이미지(Frame)들이 저장된 폴더.
- **`2_KEYPOINTS/`**: Sapiens 모델을 통해 추출한 관절 좌표 데이터 폴더. SAM 데이터와 매칭되어 객체 ID 정보가 함께 포함됨.
- **`3_MP4/`**: `2_KEYPOINTS`의 좌표 정보를 시각화하여 생성한 검토용 MP4 영상 폴더.
- **`4_INTERP_DATA/`**: 결측치 보간 및 이상치 제거 등 사후 처리(Post-processing)를 완료한 정제 데이터 폴더.
- **`5_YOLO_TXT/`**: `4_INTERP_DATA`를 YOLO 학습 형식에 맞춰 변환한 `.txt` 라벨 파일 모음.
- **`6_YOLO_TRAINING_DATA/`**: 실제 모델 학습에 즉시 사용 가능한 형태로 경로 및 데이터셋 구조가 정리된 폴더.
- **`7_INTERP_MP4/`**: 보정된 데이터(`4_INTERP_DATA`)를 바탕으로 다시 렌더링한 최종 확인용 MP4 영상 폴더.
- **`8_SAM/`**: Segment Anything Model(SAM)을 통해 추출한 객체 마스크(Mask) 및 ID 데이터 폴더.
- **`checkpoints/`**: 사전 학습 모델(Pre-trained) 또는 파인튜닝이 완료된 모델 가중치 파일이 저장된 폴더.
- **`test/`**: 실험적인 코드를 실행하고 결과의 정확도를 확인하기 위한 테스트용 폴더.
- **`walking_data/`**: 추가 확보된 보행 분석용 영상 2개에 대한 분석 결과 및 라벨링 데이터가 포함된 폴더.
- **`.csv`**: 데이터셋의 전체 이력 및 세그멘테이션 수치를 기록한 메타데이터 파일들 (v1.0 ~ v2.1).

---

## 3. Mini-Project 실험 및 결과

### 3.1. GT Pipeline (v1.0)
* **Project Link:** https://github.com/KJH121212/ASAN_01_mini_sapiens_labeling.git
* **프로젝트 목적:** Sapiens 모델을 활용하여 대규모 비디오 데이터셋에서 2D Keypoints 추출하고, 시각화(Overlay) 영상을 자동으로 생성하는 파이프라인 완성. 가장 성능이 좋은 sapiens 활용 방법 탐색. 추후 최종 GT pipeline에서 일부 코드를 차용했다.
* **실험:** 1. Sapiens 속도 향상 실험. sapiens는 파운데이션 모델인 만큼 YOLO에 비해 그 속도가 느리다. 그 느린 속도의 이유는 매 프레임 Detection을 진행을 하는 작업이 진행되기 때문이라고 판단하였고, 재활운동 비디오 특성상, 프레임당 bbox의 움직임이 거의 없다는 가정 아래, 5frame 마다 Detection을 진행하고, 나머지 frame은 이전 frame의 bbox 정보를 복사하여 사용하는 것으로 변경. 해당 방식을 통해서 속도 약 20% 향상. 정확도에는 큰 변화가 없었지만, 교수님의 속도보다는 정확도가 더 중요하다는 주장에 의해 해당 방식의 추출 방식은 채택되지 않음. 따라서 GT pipeline에 사용되지 않는 코드라고 판단하여, 코드를 삭제함.

### 3.2. SAM (Segment Anything Model) 활용
* **Project Link:** https://github.com/KJH121212/ASAN_01_mini_SAM3
* **프로젝트 목적:** Sapiens에는 존재하지 않는 Tracking에 가장 강한 SAM을 활용하여 Sapiens를 통해 추출한 skeleton에 고유 ID 부여.
* **실험:**
    - **1. 긴 영상에서의 SAM 활용:** 기본적으로 제공되는 SAM 코드의 경우, 중간에서의 Interaction을 염두에 두고 만들기 때문에, 모든 프레임의 feature 정보를 memory bank에 저장한다. 이에 따라 긴 영상의 경우 모든 프레임의 Feature map이 GPU에 저장되면 OOM(Out Of Memory) 문제가 발생하며 이를 해결하는 방법을 찾아야 한다. 해당 프로젝트에서 300 프레임씩 window shift 하는 방식을 통해 영상을 나누어서 Tracking하는 방식으로 해당 문제를 해결하였다.
    - **2. SAM3의 문자 입력을 통한 Segmentation:** SAM3에서 새롭게 생긴 능력인 text prompt 입력을 통한 segmentation을 통해 환자, 의료진 등을 구별하는 방법을 실험해 보았다. 아쉽게도 person을 입력하였을 때 높은 정확도로 해당 객체를 잘 찾아내지만, 환자와 의료진의 차이를 잘 구분해내지 못하는 모습을 보였다. 이에 따라 최종 pipeline에서는 모든 'person'을 추적하게 하고, 환자의 ID만 따로 체크를 하는 방식으로 해당 문제를 해결하였다.
    - **3. SAM to Sapiens 실험:** Sapiens는 Top-Down 방식의 Pose estimation 방식이다. 따라서 Detection에서 제공하는 Boundary Box를 SAM이 추출한 mask를 활용한다면, 높은 정확도로 추적 대상의 Skeleton을 추출할 수 있을 것이다. 실험 결과 해당 방식으로 얻은 skeleton은 Sapiens만 활용하였을 경우보다 jitter 문제가 심한 것이 확인되었다. 그러나 기존보다 높은 Detection 성능 때문에, 기존 Ground Truth 모델을 통해 추출하는데 실패한 video들을 해당 방식으로 Skeleton을 추출하였고, 모든 비디오의 skeleton을 추출하는데 성공하였다.

### 3.3. Real-time YOLO (Real Time Counter Prototype)
* **Project Link:** https://github.com/KJH121212/ASAN_01_real_time_YOLO.git
* **실험:**
    - **1. V1.0:** prototype v1.0의 경우, 특정 keypoints들이 탐지되지 않아도 keypoints를 유추하여 counting을 시도하였다. 이에 따라 Counting이 정확하지 않았다.
    - **2. V1.1:** Nomalization 하는데 필수적으로 필요한 골반이 나오지 않을 경우 뒤로 물러나라는 메시지 내보내고 count 정지. Kalman filter 추가하여 Outlier 걸러내기. 우측에 Normalization 한 skeleton 보이기.
    - **3. V1.2:** 실시간 영상 우측에 새로운 창을 생성하여 가시성 확보. Bar 업데이트(Bar에 Flex 상태와 Relax 상태의 threshold를 추가하여 게임성 확보, Relax 상태를 아래로 하도록 고정).
    - **4. V1.3:** 우측 norm된 skeleton 제거, bar와 count, FPS만 표시하도록 변경. 구동 환경을 프로젝트 폴더 내부에 생성할 수 있도록 하여 노트북이 변경되었을 경우에도 구동이 가능하도록 수정. Pytorch를 이용해 Nvidia GPU를 사용할 수 있도록 변경하였으나 GPU가 달린 노트북이 없어서 실험 실패.

### 3.4. Outlier 분석 및 필터링 전략
* **Project Link:** (소실됨)
* **Outlier 발생 원인 분석:** 라벨링 결과에서 일부 프레임에서 Outlier가 발생하는 현상을 확인할 수 있었다. 이러한 Outlier는 주로 Occlusion 문제, 좌·우(L–R) 혼동 문제, 흔들림(Blur) 문제, 환자복에 의한 관절 위치 혼동 등의 이유로 발생한다.
* **Outlier Filter 기술 비교 실험:** Confidence Score, IQR, Velocity, Angle, Kalman, Isolation Forest, FFT 등 다양한 기법에 대해 장단점을 분석하였음.
* **핵심 인사이트 및 해결 방안:** 실시간성이 높으면서 Outlier를 정확하게 탐지한 Kalman filter의 유효성을 확인했음. 그러나 범용적인 threshold를 찾아내는데 어려움을 겪어 Skeleton Normalization을 조사함. 최종적으로 Confidence score < 0.07 기준과 Kalman filter를 병합 활용함.

---

## 4. Ground Truth (GT) Pipeline

![GT Pipeline](./img/GT_pipeline.png)

### **Step 1: Frame Extraction**
* **파일:** `./runner/step1.py`
* **설명:** 원본 영상을 분석 효율을 위해 정해진 해상도(720p)로 리사이징하며 프레임 단위 이미지로 추출합니다.

### **Step 2: Skeleton Extraction (Initial)**
* **파일:** `./runner/step2_sapiens.py`
* **설명:** RTMDet으로 탐지된 Bounding Box 정보를 입력값으로 하여, Sapiens 모델을 통해 객체의 Skeleton 정보를 1차 추출합니다.

### **Step 3: Mask & ID Tracking**
* **파일:** `./runner/step3_sam.py`
* **설명:** SAM을 활용하여 'person' 텍스트 프롬프트를 입력하여 해당 프레임의 모든 사람을 segmentation합니다. 이후 프레임 간 추적(Tracking)을 통해 객체별 고유 ID를 부여하고 정밀 Mask 데이터를 추출합니다.

### **Step 3.5: Refined Skeleton with Mask**
* **파일:** `./runner/step3.5_sapiens_with_sam.py`
* **설명:** 앞 단계에서 확보한 SAM의 Mask 데이터를 가이드로 활용하여, Sapiens 모델이 더욱 정교한 Skeleton 데이터를 생성하도록 유도합니다.

### **Step 4: ID Assignment & Integration**
* **파일:** `./runner/step4_assign_ids.py`
* **설명:** Sapiens로 추출된 Skeleton 데이터에 SAM의 고유 ID를 매칭하여 최종적인 객체 식별 Skeleton 데이터를 완성합니다.

### **Postprocessing**
* **파일:** `./runner/postprocessing_filter.py`
* **설명:** kalman filter, IQR, velocity 등 다양한 기법을 활용하여 skeleton의 Outlier들을 필터링 하여 YOLO finetuning에 활용할 수 있는 데이터를 `interp_data` 폴더에 따로 저장.