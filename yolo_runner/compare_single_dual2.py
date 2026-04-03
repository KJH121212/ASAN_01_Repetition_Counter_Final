import sys
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# =================================================================
# 0. 유틸리티 함수 (Letterbox 및 좌표 역산)
# =================================================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]     # 현재 이미지의 높이(H)와 너비(W)를 가져옵니다.
    
    # 목표 모양(new_shape)이 정수(예: 640)라면 (640, 640) 형태의 튜플로 변환합니다.
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        
    # 원본 대비 목표 크기의 가로/세로 비율 중 더 작은 값을 선택합니다 (이미지가 잘리지 않게 하기 위함).
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 선택된 비율(r)을 적용했을 때의 새로운 너비와 높이를 계산합니다 (반올림 처리).
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # 목표 크기에서 리사이즈된 크기를 뺀 나머지 여백(Padding) 길이를 계산합니다.
    # 2로 나누어 상/하 또는 좌/우 양쪽에 균등하게 배분할 준비를 합니다.
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    
    # 원본 비율을 유지한 채 리사이즈가 필요한 경우(현재 크기와 계산된 크기가 다를 때) OpenCV로 리사이즈합니다.
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 상하 여백(dh)을 정수로 변환합니다. (0.1을 빼는 것은 부동소수점 오차로 인한 크기 초과를 방지하는 트릭입니다.)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    
    # 좌우 여백(dw)을 정수로 변환합니다.
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 이미지 주변에 계산된 여백(Border)을 추가하고, 이미지, 적용된 비율, 여백 정보를 반환합니다.
    # (이후 좌표 복구를 위해 r, dw, dh 값이 필요합니다.)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)

def get_padded_crop(img, bbox, pad_ratio=0.2):
    """
    bbox: [x1, y1, x2, y2]
    pad_ratio: 박스 크기 대비 추가할 여백 비율 (0.2 = 20%)
    """
    h_orig, w_orig = img.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # 1. 원래 박스의 너비와 높이 계산
    w, h = x2 - x1, y2 - y1
    
    # 2. 여백 크기 계산 (가로/세로 각각 적용)
    dw, dh = w * pad_ratio, h * pad_ratio
    
    # 3. 여백을 포함한 새로운 좌표 계산 (이미지 경계를 벗어나지 않도록 제한)
    nx1 = max(0, int(x1 - dw))
    ny1 = max(0, int(y1 - dh))
    nx2 = min(w_orig, int(x2 + dw))
    ny2 = min(h_orig, int(y2 + dh))
    
    # 4. Crop 수행 및 적용된 오프셋 반환 (나중에 좌표 복구 시 필요)
    crop = img[ny1:ny2, nx1:nx2]
    return crop, (nx1, ny1)

def scale_coords_back(kpts, ratio, pad, offset):
    # kpts: [17, 2] tensor or numpy
    kpts[:, 0] -= pad[0]
    kpts[:, 1] -= pad[1]
    kpts[:, :2] /= ratio
    kpts[:, 0] += offset[0]
    kpts[:, 1] += offset[1]
    return kpts

# =================================================================
# 1. 17 Keypoints (COCO) 뼈대 연결 정의
# =================================================================
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]
import json

def draw_gt_from_json(img, json_path):
    out_img = img.copy()
    if not json_path.exists():
        return out_img

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 메타 정보에서 스켈레톤 연결 고리 가져오기
    links = data['meta_info']['skeleton_links']
    
    for instance in data['instance_info']:
        kpts = np.array(instance['keypoints']) # [17, 2] (x, y)
        scores = np.array(instance['keypoint_scores']) # [17]
        
        # 뼈대(Links) 그리기
        for p1_idx, p2_idx in links:
            # JSON은 0-based 인덱스이므로 그대로 사용
            if scores[p1_idx] > 0.00 and scores[p2_idx] > 0.0: # 가시성 확인
                pt1 = tuple(map(int, kpts[p1_idx]))
                pt2 = tuple(map(int, kpts[p2_idx]))
                cv2.line(out_img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 관절(Points) 그리기
        for i, (x, y) in enumerate(kpts):
            if scores[i] > 0.0:
                cv2.circle(out_img, (int(x), int(y)), 4, (255, 0, 0), -1)
                
        # Bbox 그리기 (선택 사항)
        # bx1, by1, bx2, by2 = map(int, instance['bbox'])
        # cv2.rectangle(out_img, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

    return out_img

def draw_skeleton_17(img, kpts, color=(0, 255, 0), thickness=2):
    # kpts: [17, 3] (x, y, conf)
    for p1, p2 in SKELETON_17:
        if p1 < len(kpts) and p2 < len(kpts):
            x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
            x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
            conf1 = kpts[p1][2] if len(kpts[p1]) > 2 else 1.0
            conf2 = kpts[p2][2] if len(kpts[p2]) > 2 else 1.0
            
            if conf1 > 0.25 and conf2 > 0.25: # 신뢰도 임계값
                cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    
    # 관절 점 그리기
    for x, y, conf in kpts:
        if conf > 0.25:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
    return img

# =================================================================
# 2. 경로 설정 및 모델 로드
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))
from utils.path_list import path_list

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("🔄 모델 로드 중...")
det_model = YOLO('yolo11n.pt').to(device)
pose_model = YOLO(DATA_DIR / "checkpoints/YOLO/yolo11n-pose.pt").to(device)

# 샘플 데이터 1개 선정 (메타데이터 기준 첫 번째)
target_df = pd.read_csv(DATA_DIR / "metadata_v2.1.csv").iloc[[22]] 

for idx, row in target_df.iterrows():
    cp = row['common_path']
    paths = path_list(cp)
    frame_dir = Path(paths['frame'])
    frame_paths = sorted(list(frame_dir.glob("*.jpg")))
    
    if not frame_paths: continue
    
    # 해상도 설정
    first_img = cv2.imread(str(frame_paths[0]))
    h, w = first_img.shape[:2]
    cell_w, cell_h = w // 2, h // 2 # 4분할 시 각 셀의 크기
    
    out_name = f'./compare_17kpt_split_{idx}.mp4'
    video_out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

    print(f"🎬 비디오 생성 시작: {out_name}")
    for f_path in tqdm(frame_paths):
        img = cv2.imread(str(f_path))
        if img is None: continue

        # --- 1. RAW (좌상) ---
        v_raw = cv2.resize(img.copy(), (cell_w, cell_h))
        cv2.putText(v_raw, "1. RAW", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # --- 2. GT (우상) - 초록색 ---
        v_gt = img.copy()
        json_path = Path(paths['interp_data']) / f"{f_path.stem}.json" # 2. GT (우상) - JSON 데이터 기반 렌더링
        v_gt = draw_gt_from_json(img, json_path)
        v_gt = cv2.resize(v_gt, (cell_w, cell_h))
        cv2.putText(v_gt, "2. GT (from JSON)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # --- 3. Single-Stage (좌하) - 빨간색 ---
        res_s = pose_model.predict(img, verbose=False, device=device)[0]
        v_s = img.copy()
        if res_s.keypoints is not None and len(res_s.keypoints.data) > 0:
            kpts_s = res_s.keypoints.data[0].cpu().numpy() # [17, 3]
            v_s = draw_skeleton_17(v_s, kpts_s, color=(0, 0, 255))
        v_s = cv2.resize(v_s, (cell_w, cell_h))
        cv2.putText(v_s, "3. SINGLE-STAGE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
# --- 4. Double-Stage (우하) - 노란색 (Padding 적용) ---
        v_d = img.copy()
        res_d = det_model.predict(img, classes=[0], verbose=False, device=device)[0]
        
        if len(res_d.boxes) > 0:
            # 1단계: 가장 높은 점수의 사람 Bbox 추출
            box = res_d.boxes[0].xyxy[0].cpu().numpy().astype(int)
            
            # 💡 2단계: 20% 여백을 추가하여 Crop 수행 (nx1, ny1은 여백 포함 시작점)
            crop, (nx1, ny1) = get_padded_crop(img, box, pad_ratio=0.2)
            
            if crop.size > 0:
                # 3단계: Letterbox 및 Pose 추론
                c_img, r, p = letterbox(crop, 640)
                res_p = pose_model.predict(c_img, verbose=False, device=device)[0]
                
                if res_p.keypoints is not None and len(res_p.keypoints.data) > 0:
                    # 💡 4단계: 예측값을 복제한 후 원본 좌표계로 역산 (nx1, ny1 반영)
                    kpts_p = res_p.keypoints.data[0].clone()
                    kpts_p = scale_coords_back(kpts_p, r, p, (nx1, ny1))
                    
                    # 5단계: 노란색(0, 255, 255)으로 뼈대 그리기
                    v_d = draw_skeleton_17(v_d, kpts_p.cpu().numpy(), color=(0, 255, 255))
        
        v_d = cv2.resize(v_d, (cell_w, cell_h))
        cv2.putText(v_d, "4. DOUBLE-STAGE (PAD-CROP)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # 4분할 병합
        top = np.hstack([v_raw, v_gt])
        bottom = np.hstack([v_s, v_d])
        combined = np.vstack([top, bottom])
        
        video_out.write(combined)

    video_out.release()
    print(f"✅ 완료: {out_name}")