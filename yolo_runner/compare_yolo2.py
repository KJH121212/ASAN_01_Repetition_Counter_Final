import sys # 시스템 설정을 위한 모듈입니다.
import numpy as np # 배열 병합 및 오차 계산을 위한 수치 연산 라이브러리입니다.
import pandas as pd # 메타데이터 데이터프레임을 다루기 위한 판다스입니다.
import cv2 # 비디오 생성, 고화질 리사이징, 커스텀 선 그리기를 위한 OpenCV입니다.
from pathlib import Path # 안전한 경로 탐색을 위한 모듈입니다.
from ultralytics import YOLO # YOLO 추론을 위한 라이브러리입니다.
from tqdm import tqdm # 진행 상황을 보여주는 진행률 바입니다.

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/") 
sys.path.append(str(BASE_DIR)) 
from utils.path_list import path_list 

# =================================================================
# 1. 뼈대(Skeleton) 연결 인덱스 정의 (매우 중요!)
# =================================================================
# 17 Keypoint (COCO 기본 규격 - 정답지 및 사전 학습 모델용)
SKELETON_17 = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
        (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
]

# 12 Keypoint (파인튜닝된 커스텀 모델용: 얼굴 5개 제외하고 어깨부터 발목까지)
# 인덱스 가정: 0:왼어깨, 1:오른어깨, 2:왼팔꿈치, 3:오른팔꿈치, 4:왼손목, 5:오른손목, 
#              6:왼골반, 7:오른골반, 8:왼무릎, 9:오른무릎, 10:왼발목, 11:오른발목
SKELETON_12 = [
    (0, 1),   # 어깨 연결
    (0, 2), (2, 4), # 왼쪽 팔
    (1, 3), (3, 5), # 오른쪽 팔
    (0, 6), (1, 7), # 몸통 (어깨-골반)
    (6, 7),   # 골반 연결
    (6, 8), (8, 10), # 왼쪽 다리
    (7, 9), (9, 11)  # 오른쪽 다리
]

# =================================================================
# 2. 경로 설정 및 대상 추출 (3개 샘플링)
# =================================================================
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data") 
METADATA1_PATH = DATA_DIR / "metadata_v2.0.csv" 
METADATA2_PATH = DATA_DIR / "metadata_v2.1.csv" 

def get_changed_data(p1, p2): 
    df1 = pd.read_csv(p1) 
    df2 = pd.read_csv(p2) 
    merged = pd.merge(df2, df1[['common_path', 'is_train', 'is_val']], on='common_path', how='left', suffixes=('', '_old')) 
    condition = (((merged['is_train'] == True) & (merged['is_train_old'] != True)) | 
                 ((merged['is_val'] == True) & (merged['is_val_old'] != True))) 
    return merged[condition].reset_index(drop=True) 

target_df = get_changed_data(METADATA1_PATH, METADATA2_PATH) # 3개만 샘플링
# 0, 6, 24번 행만 선택 (기존 target_df 갱신)
target_df = target_df.iloc[[0, 11, 24]].reset_index(drop=True)

print(f"✅ 분석 대상 비디오 개수: {len(target_df)}개 (고화질 3x3 뼈대 연결 테스트)")

# =================================================================
# 3. 모델 로드 
# =================================================================
print("\n🔄 평가할 YOLO 모델 7개를 메모리에 로드 중입니다...") 
models = { 
    "PRE_MODEL": YOLO(DATA_DIR / "checkpoints/YOLO/yolo11n-pose.pt"), 
    "V1.0_STEP1": YOLO(DATA_DIR / "checkpoints/YOLO_FINETUNING/v1.0_step1/weights/best.pt"), 
    "V1.0_STEP15": YOLO(DATA_DIR / "checkpoints/YOLO_FINETUNING/v1.0_step15/weights/best.pt"), 
    "V1.0_STEP30": YOLO(DATA_DIR / "checkpoints/YOLO_FINETUNING/v1.0_step30/weights/best.pt"), 
    "V2.0_STEP1": YOLO(DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step1/weights/best.pt"), 
    "V2.0_STEP10": YOLO(DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step10/weights/best.pt"), 
    "V2.0_STEP30": YOLO(DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step30/weights/best.pt") 
} 

# =================================================================
# 4. 고화질 렌더링 & 뼈대(Skeleton) 그리기 도우미 함수들
# =================================================================
def draw_custom_skeleton(img, kpts, skeleton_links, line_color=(255, 100, 100), point_color=(0, 0, 255)):
    # 뼈대 선(Line) 그리기
    for p1, p2 in skeleton_links:
        if p1 < len(kpts) and p2 < len(kpts): # 인덱스가 안전한지 확인
            x1, y1, v1 = kpts[p1]
            x2, y2, v2 = kpts[p2]
            if v1 > 0.3 and v2 > 0.3: # 양쪽 관절이 모두 어느 정도 신뢰도(가시성)가 있을 때만 선을 잇습니다.
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 4, cv2.LINE_AA)
    # 관절 점(Point) 그리기
    for x, y, v in kpts:
        if v > 0.3:
            cv2.circle(img, (int(x), int(y)), 7, point_color, -1, cv2.LINE_AA)
    return img

def draw_prediction(img_raw, res, skeleton_type=12):
    out_img = img_raw.copy()
    if res.boxes is not None and len(res.boxes.xyxy) > 0:
        # 박스 그리기
        x1, y1, x2, y2 = map(int, res.boxes.xyxy[0].cpu().numpy())
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 200, 0), 3) # 하늘색 박스
        
        # 키포인트 뼈대 그리기
        if res.keypoints is not None and len(res.keypoints.data) > 0:
            kpts = res.keypoints.data[0].cpu().numpy() # [x, y, conf] 
            links = SKELETON_12 if skeleton_type == 12 else SKELETON_12
            draw_custom_skeleton(out_img, kpts, links, line_color=(0, 255, 255), point_color=(0, 0, 255))
    return out_img

def draw_target(img, txt_path): 
    out_img = img.copy() 
    h, w = out_img.shape[:2] 
    
    if txt_path.exists(): 
        with open(txt_path, 'r') as f: lines = f.readlines() 
        if lines: 
            parts = list(map(float, lines[0].strip().split())) 
            cx, cy = int(parts[1] * w), int(parts[2] * h) 
            bw, bh = int(parts[3] * w), int(parts[4] * h) 
            cv2.rectangle(out_img, (cx - bw//2, cy - bh//2), (cx + bw//2, cy + bh//2), (0, 255, 0), 4) 
            
            kpts = []
            for i in range(5, len(parts), 3): 
                kpts.append([parts[i] * w, parts[i+1] * h, parts[i+2]])
            draw_custom_skeleton(out_img, kpts, SKELETON_12, line_color=(0, 255, 0), point_color=(255, 0, 0)) # 초록선, 파란점
    return out_img 

def add_label(img, label_text): 
    cv2.putText(img, label_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA) 
    return img 

def resize_hq(img, size): 
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) 

# =================================================================
# 5. 3x3 그리드 비디오 고화질 병합 루프
# =================================================================
SCALE_FACTOR = 0.5 # 2.5K 화질 유지 (원하시면 조절 가능)

for idx, row in target_df.iterrows(): 
    cp = row['common_path'] 
    paths_dict = path_list(cp) 
    frame_dir = Path(paths_dict['frame']) 
    
    frame_paths = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))) 
    if not frame_paths: continue 
        
    first_frame = cv2.imread(str(frame_paths[0])) 
    h, w = first_frame.shape[:2] 
    
    cell_w, cell_h = int(w * SCALE_FACTOR), int(h * SCALE_FACTOR) 
    grid_w, grid_h = cell_w * 3, cell_h * 3 
    
    out_name = f"./img/yolo_compare_grid_HQ_{idx+1}.mp4" 
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (grid_w, grid_h)) 
    
    print(f"\n🎬 {idx+1}번째 고화질 3x3 비디오 생성 중... 해상도: {grid_w}x{grid_h}") 
    for f_path in tqdm(frame_paths, desc="프레임 병합 중"): 
        txt_path = Path(paths_dict.get('yolo_txt', '')) / f"{f_path.stem}.txt" 
        if not txt_path.exists(): txt_path = Path(paths_dict['interp_data']) / f"{f_path.stem}.txt" 
            
        img_raw = cv2.imread(str(f_path)) 
        if img_raw is None: continue 
        
        # --- 1행 생성 ---
        cell_raw = add_label(resize_hq(img_raw.copy(), (cell_w, cell_h)), "RAW") 
        cell_target = add_label(resize_hq(draw_target(img_raw, txt_path), (cell_w, cell_h)), "TARGET(GT)") 
        
        # 모델 예측 후 커스텀 함수로 뼈대 렌더링 (PRE_MODEL은 17kpt)
        res_pre = models["PRE_MODEL"].predict(img_raw.copy(), verbose=False)[0]
        cell_pre = add_label(resize_hq(draw_prediction(img_raw.copy(), res_pre, skeleton_type=17), (cell_w, cell_h)), "PRE_MODEL") 
        row1 = np.hstack([cell_raw, cell_target, cell_pre]) 
        
        # --- 2행 생성 (파인튜닝 모델은 12kpt) ---
        res_v1_1 = models["V1.0_STEP1"].predict(img_raw.copy(), verbose=False)[0]
        cell_v1_1 = add_label(resize_hq(draw_prediction(img_raw.copy(), res_v1_1, skeleton_type=12), (cell_w, cell_h)), "V1.0_STEP1") 
        res_v1_15 = models["V1.0_STEP15"].predict(img_raw.copy(), verbose=False)[0]
        cell_v1_15 = add_label(resize_hq(draw_prediction(img_raw.copy(), res_v1_15, skeleton_type=12), (cell_w, cell_h)), "V1.0_STEP15") 
        res_v1_30 = models["V1.0_STEP30"].predict(img_raw.copy(), verbose=False)[0]
        cell_v1_30 = add_label(resize_hq(draw_prediction(img_raw.copy(), res_v1_30, skeleton_type=12), (cell_w, cell_h)), "V1.0_STEP30") 
        row2 = np.hstack([cell_v1_1, cell_v1_15, cell_v1_30]) 
        
        # --- 3행 생성 (파인튜닝 모델은 12kpt) ---
        res_v2_1 = models["V2.0_STEP1"].predict(img_raw.copy(), verbose=False)[0]
        cell_v2_1 = add_label(resize_hq(draw_prediction(img_raw.copy(), res_v2_1, skeleton_type=12), (cell_w, cell_h)), "V2.0_STEP1") 
        res_v2_10 = models["V2.0_STEP10"].predict(img_raw.copy(), verbose=False)[0]
        cell_v2_10 = add_label(resize_hq(draw_prediction(img_raw.copy(), res_v2_10, skeleton_type=12), (cell_w, cell_h)), "V2.0_STEP10") 
        res_v2_30 = models["V2.0_STEP30"].predict(img_raw.copy(), verbose=False)[0]
        cell_v2_30 = add_label(resize_hq(draw_prediction(img_raw.copy(), res_v2_30, skeleton_type=12), (cell_w, cell_h)), "V2.0_STEP30") 
        row3 = np.hstack([cell_v2_1, cell_v2_10, cell_v2_30]) 
        
        # --- 3x3 수직 병합 및 쓰기 ---
        grid_frame = np.vstack([row1, row2, row3]) 
        out.write(grid_frame) 
        
    out.release() 
    print(f"🎉 고화질 뼈대 연결 영상 저장 완료: {out_name}")