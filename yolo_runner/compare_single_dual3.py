import cv2
import pandas as pd
import torch
import sys
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

def extract_detections_to_df(video_path, model_path='yolo11n.pt', device='cuda:0'):
    # 1. 모델 로드 및 비디오 캡처 설정
    model = YOLO(model_path).to(device) # 지정된 장치(GPU/CPU)로 모델을 로드합니다.
    cap = cv2.VideoCapture(str(video_path)) # 비디오 파일을 읽기 위한 객체를 생성합니다.
    
    # 비디오의 전체 프레임 수를 가져와서 진행바(tqdm)에 사용합니다.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    fps = cap.get(cv2.CAP_PROP_FPS) # 비디오의 FPS 정보를 저장합니다.
    
    detection_data = [] # 프레임별 데이터를 저장할 리스트입니다.

    print(f"🚀 분석 시작: {video_path.name} ({total_frames} frames)")

    # 2. 프레임별 루프 실행
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read() # 한 프레임을 읽어옵니다.
        if not ret: # 프레임이 없으면 루프를 종료합니다.
            break

        # YOLO 추론 수행 (사람만 탐지하려면 classes=[0] 설정)
        results = model.predict(frame, verbose=False, device=device, classes=[0])[0]

        # 3. 탐지된 결과 추출
        if len(results.boxes) > 0:
            for box in results.boxes:
                # 좌표(xyxy), 신뢰도(conf), 클래스(cls)를 넘파이 형태로 가져옵니다.
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # 박스 좌표를 추출합니다.
                conf = box.conf[0].cpu().item() # 신뢰도 점수를 추출합니다.
                cls = box.cls[0].cpu().item() # 클래스 번호를 추출합니다.

                # 데이터 리스트에 사전 형태로 추가합니다.
                detection_data.append({
                    'frame': frame_idx, # 현재 프레임 번호
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, # 바운딩 박스 좌표
                    'confidence': conf, # 탐지 확신도
                    'class': int(cls), # 탐지된 객체 종류
                    'timestamp': frame_idx / fps # 시간 정보(초)를 계산하여 저장합니다.
                })
        else:
            # 탐지된 객체가 없는 경우에도 프레임 정보는 남겨두는 것이 분석에 유리합니다.
            detection_data.append({
                'frame': frame_idx,
                'x1': None, 'y1': None, 'x2': None, 'y2': None,
                'confidence': 0,
                'class': None,
                'timestamp': frame_idx / fps
            })

    cap.release() # 비디오 자원을 해제합니다.
    
    # 4. 리스트를 DataFrame으로 변환
    df = pd.DataFrame(detection_data) # 수집된 데이터를 판다스 데이터프레임으로 만듭니다.
    return df # 생성된 데이터프레임을 반환합니다.

import cv2
import pandas as pd
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

def extract_pose_from_frames(frame_dir, model_path='yolo11n-pose.pt', device='cuda:0'):
    # 1. 모델 로드 및 이미지 파일 리스트 확보
    model = YOLO(model_path).to(device) # Pose 모델을 지정된 장치에 로드합니다.
    frame_dir = Path(frame_dir) # 문자열 경로를 Path 객체로 변환합니다.
    
    # 해당 폴더에서 jpg, png 등 이미지 확장자를 가진 파일만 골라 정렬합니다.
    extensions = ("*.jpg", "*.jpeg", "*.png") # 처리할 이미지 확장자 목록입니다.
    frame_paths = [] # 이미지 파일 경로들을 담을 리스트입니다.
    for ext in extensions:
        frame_paths.extend(list(frame_dir.glob(ext))) # 확장자별로 파일을 찾아 추가합니다.
    
    frame_paths = sorted(frame_paths) # 파일명 순으로 정렬하여 시간 순서를 맞춥니다.
    
    if not frame_paths: # 읽어올 이미지가 없다면 에러 메시지를 출력합니다.
        print(f"❌ '{frame_dir}' 경로에 이미지 파일이 없습니다.")
        return pd.DataFrame()

    # COCO 17개 관절 명칭 정의
    keypoint_names = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"
    ]
    
    pose_data = [] # 추출된 데이터를 저장할 리스트입니다.

    print(f"🖼️ 프레임 분석 시작: {len(frame_paths)} 장 발견")

    # 2. 이미지 리스트 루프 실행
    for idx, f_path in enumerate(tqdm(frame_paths)):
        img = cv2.imread(str(f_path)) # 이미지를 읽어옵니다.
        if img is None: # 이미지를 읽지 못했을 경우 건너뜁니다.
            continue

        # YOLO Pose 추론 수행
        results = model.predict(img, verbose=False, device=device)[0]

        # 기본 행 데이터 구성 (프레임 인덱스와 파일명 포함)
        row = {
            'frame_idx': idx, # 순차적인 프레임 번호입니다.
            'file_name': f_path.name # 나중에 추적하기 위해 파일명을 기록합니다.
        }

        # 3. 키포인트 데이터 추출
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            # 첫 번째 탐지된 사람의 데이터를 가져옵니다.
            kpts = results.keypoints.data[0].cpu().numpy() # [17, 3] (x, y, conf)
            
            for i, name in enumerate(keypoint_names):
                row[f'{name}_x'] = kpts[i][0] # x좌표 저장
                row[f'{name}_y'] = kpts[i][1] # y좌표 저장
                row[f'{name}_conf'] = kpts[i][2] # 신뢰도 저장
        else:
            # 탐지 실패 시 None 및 0으로 채웁니다.
            for name in keypoint_names:
                row[f'{name}_x'], row[f'{name}_y'], row[f'{name}_conf'] = None, None, 0
        
        pose_data.append(row)

    # 4. DataFrame 변환 및 결과 반환
    df = pd.DataFrame(pose_data) # 전체 리스트를 데이터프레임으로 변환합니다.
    return df # 최종 데이터프레임을 반환합니다.


# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from utils.path_list import path_list

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data_v2.1.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================
print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSANJIN_PATH.name})")
meta_df = pd.read_csv(METADATA_PATH)
bosan_df = pd.read_csv(BOSANJIN_PATH)

target = 83

# metadata  사용시
common_path = meta_df.iloc[target]['common_path']
start_frame = 0
end_frame = None

paths = path_list(common_path)
patient_id = int(meta_df.loc[meta_df['common_path'] == common_path, 'patient_id'].iloc[0]) # 조건에 맞는 첫 번째 ID 값을 안전하게 꺼내어 정수형(int)으로 변환합니다.


# --- 사용 예시 ---

df_pose_frames = extract_pose_from_frames(paths['frame'])
print(df_pose_frames.head())

# --- 사용 예시 ---
# df_pose = extract_pose_to_df(Path("exercise_video.mp4"))
# print(df_pose[["frame", "nose_x", "nose_y", "l_shoulder_x"]].head()) # 일부 데이터 확인
# --- 사용 예시 ---
# video_file = Path("your_video.mp4")
# df_results = extract_detections_to_df(video_file)
# df_results.to_csv("detection_results.csv", index=False) # 결과를 CSV로 저장합니다.
# print(df_results.head()) # 상위 5개 데이터를 확인합니다.