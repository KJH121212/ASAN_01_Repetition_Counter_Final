import sys
import json
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

cv2.setNumThreads(0)

# =================================================================
# 0. 공통 유틸리티 함수 (Padding, Letterbox, 좌표 복구)
# =================================================================
def get_padded_crop(img, bbox, pad_ratio=0.2):
    h_orig, w_orig = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    dw, dh = w * pad_ratio, h * pad_ratio
    nx1, ny1 = max(0, int(x1 - dw)), max(0, int(y1 - dh))
    nx2, ny2 = min(w_orig, int(x2 + dw)), min(h_orig, int(y2 + dh))
    return img[ny1:ny2, nx1:nx2], (nx1, ny1)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)

def scale_coords_back(kpts, ratio, pad, offset):
    kpts[:, 0] -= pad[0]
    kpts[:, 1] -= pad[1]
    kpts[:, :2] /= ratio
    kpts[:, 0] += offset[0]
    kpts[:, 1] += offset[1]
    return kpts

# 💡 얼굴(0~4) 제외, 신체(5~16) 연결 정의
BODY_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
    (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10)
]

def draw_skeleton_body(img, kpts, color=(0, 255, 0), thickness=2):
    """신뢰도 무시하고 무조건 5~16번 그리기"""
    for p1, p2 in BODY_LINKS:
        if p1 < len(kpts) and p2 < len(kpts):
            x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
            x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    
    for i in range(5, min(17, len(kpts))):
        x, y = int(kpts[i][0]), int(kpts[i][1])
        cv2.circle(img, (x, y), 4, color, -1)
    return img

# =================================================================
# 설정 및 경로 초기화
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))
from utils.path_list import path_list

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 12개 신체 시그마
BODY_SIGMAS = torch.tensor([
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], device=device)

# 💡 상위 5개 데이터 추출
target_df = pd.read_csv(DATA_DIR / "metadata_v2.1.csv").head(5)
print(f"✅ 분석 대상 비디오 개수: {len(target_df)}개 (상위 5개 샘플링)")

# Numpy 저장 경로
NPY_SAVE_PATH = "./extracted_kpts_top5.npy"

# =================================================================
# [STEP 1] 모델 추론 및 예측 Keypoints를 Numpy로 추출/저장
# =================================================================
print("\n" + "="*50)
print("🚀 [STEP 1] 모델 추론 및 예측 결과 Numpy 저장 시작")
print("="*50)

# 이전에 추출한 파일이 없다면 새로 추출
if not Path(NPY_SAVE_PATH).exists():
    det_model = YOLO('yolo11n.pt').to(device)
    pose_model = YOLO(DATA_DIR / "checkpoints/YOLO/yolo11n-pose.pt").to(device)
    
    all_predictions = {} # 예측 결과를 담을 딕셔너리
    
    for idx, row in target_df.iterrows():
        cp = row['common_path']
        paths = path_list(cp)
        frame_paths = sorted(list(Path(paths['frame']).glob("*.jpg")))
        
        vid_preds = {}
        for f_path in tqdm(frame_paths, desc=f"비디오 {idx+1}/5 추출 중"):
            img = cv2.imread(str(f_path))
            if img is None: continue
            
            frame_name = f_path.stem
            vid_preds[frame_name] = {'single': None, 'double': None}
            
            # 1. Single-Stage
            res_s = pose_model.predict(img, verbose=False, device=device)[0]
            if res_s.keypoints is not None and len(res_s.keypoints.xy) > 0:
                vid_preds[frame_name]['single'] = res_s.keypoints.xy[0].cpu().numpy()
                
            # 2. Double-Stage (Padding Crop)
            res_d = det_model.predict(img, classes=[0], verbose=False, device=device)[0]
            if len(res_d.boxes) > 0:
                box = res_d.boxes[0].xyxy[0].cpu().numpy().astype(int)
                crop, (nx1, ny1) = get_padded_crop(img, box, pad_ratio=0.2)
                if crop.size > 0:
                    c_img, r, p = letterbox(crop, 640)
                    res_p = pose_model.predict(c_img, verbose=False, device=device)[0]
                    if res_p.keypoints is not None and len(res_p.keypoints.xy) > 0:
                        pred_p = scale_coords_back(res_p.keypoints.xy[0].clone(), r, p, (nx1, ny1))
                        vid_preds[frame_name]['double'] = pred_p.cpu().numpy()
        
        all_predictions[cp] = vid_preds
        
    # Dictionary 형태를 포함할 수 있도록 allow_pickle=True 형태로 저장
    np.save(NPY_SAVE_PATH, all_predictions, allow_pickle=True)
    print(f"✅ 추출 완료: {NPY_SAVE_PATH} 저장됨.")
else:
    print(f"✅ 기존 추출 파일({NPY_SAVE_PATH})을 발견하여 로드합니다.")

# =================================================================
# [STEP 2] 저장된 Numpy를 불러와서 평가지표(OKS) 계산
# =================================================================
print("\n" + "="*50)
print("🚀 [STEP 2] 평가지표(OKS) 분석 (얼굴 제외 신체 12kpt)")
print("="*50)

# 저장된 Numpy 로드
saved_preds = np.load(NPY_SAVE_PATH, allow_pickle=True).item()

def calculate_oks_torch(pred_12, gt_12, scale, sigmas):
    # 모두 12개 길이의 텐서가 들어옴
    dists_sq = torch.sum((pred_12[:, :2] - gt_12[:, :2])**2, dim=1)
    v_mask = gt_12[:, 2] > 0 # 가시성 마스크
    if torch.sum(v_mask) == 0: return 0.0
    denom = 2 * (scale**2) * (sigmas**2)
    return (torch.sum(torch.exp(-dists_sq / denom)[v_mask]) / torch.sum(v_mask)).item()

metrics = {'Single-Stage': {'okss': [], 'det': 0}, 'Double-Stage': {'okss': [], 'det': 0}}
total_frames = 0

for idx, row in target_df.iterrows():
    cp = row['common_path']
    paths = path_list(cp)
    
    if cp not in saved_preds: continue
    vid_preds = saved_preds[cp]
    
    for f_path in Path(paths['frame']).glob("*.jpg"):
        frame_name = f_path.stem
        json_path = Path(paths['interp_data']) / f"{frame_name}.json"
        
        if not json_path.exists() or frame_name not in vid_preds: continue
        total_frames += 1
        
        # GT JSON 파싱
        with open(json_path, 'r') as f:
            data = json.load(f)
            if not data['instance_info']: continue
            
            # JSON은 절대 픽셀 좌표
            inst = data['instance_info'][0]
            gt_kpts = np.array([[k[0], k[1], s] for k, s in zip(inst['keypoints'], inst['keypoint_scores'])])
            
            # 이미지 원본 크기 확보를 위해 박스 스케일링
            # JSON에 bbox가 있다면 bbox 면적 활용, 없다면 이미지 읽어야 함. 
            # (속도를 위해 bbox로 처리)
            bx1, by1, bx2, by2 = inst['bbox']
            scale = np.sqrt((bx2 - bx1) * (by2 - by1))
            
            gt_t = torch.tensor(gt_kpts, device=device, dtype=torch.float32)
            
            # 1. Single-Stage 채점 (얼굴 [0~4] 제외, 5~16번 사용)
            p_single = vid_preds[frame_name]['single']
            if p_single is not None and len(p_single) >= 17 and len(gt_kpts) >= 17:
                p_t = torch.tensor(p_single, device=device, dtype=torch.float32)
                metrics['Single-Stage']['okss'].append(calculate_oks_torch(p_t[5:17], gt_t[5:17], scale, BODY_SIGMAS))
                metrics['Single-Stage']['det'] += 1
                
            # 2. Double-Stage 채점 (얼굴 제외)
            p_double = vid_preds[frame_name]['double']
            if p_double is not None and len(p_double) >= 17 and len(gt_kpts) >= 17:
                p_t = torch.tensor(p_double, device=device, dtype=torch.float32)
                metrics['Double-Stage']['okss'].append(calculate_oks_torch(p_t[5:17], gt_t[5:17], scale, BODY_SIGMAS))
                metrics['Double-Stage']['det'] += 1

report = pd.DataFrame([{
    "Strategy": k, 
    "Det Rate(%)": round(v['det']/total_frames*100, 2) if total_frames else 0,
    "Mean OKS": round(np.mean(v['okss']), 4) if v['okss'] else 0
} for k, v in metrics.items()])

print("\n📊 [포즈 추정 12kpt 비교 리포트 (상위 5개 비디오)]")
print(report.to_string(index=False))

# =================================================================
# [STEP 3] 추출된 Numpy를 활용하여 "마지막 1개" 영상 4분할 시각화
# =================================================================
last_idx = target_df.index[-1]
last_row = target_df.iloc[-1]
last_cp = last_row['common_path']
print("\n" + "="*50)
print(f"🚀 [STEP 3] 마지막 1개 비디오(Index: {last_idx}) 4분할 시각화 렌더링")
print("="*50)

paths = path_list(last_cp)
frame_paths = sorted(list(Path(paths['frame']).glob("*.jpg")))
vid_preds = saved_preds.get(last_cp, {})

h, w = cv2.imread(str(frame_paths[0])).shape[:2]
cell_w, cell_h = w // 2, h // 2
out_name = f'./compare_top5_last_video.mp4'
video_out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

for f_path in tqdm(frame_paths, desc="비디오 렌더링 중"):
    img = cv2.imread(str(f_path))
    if img is None: continue
    frame_name = f_path.stem

    # --- 1. RAW (좌상) ---
    v_raw = cv2.resize(img.copy(), (cell_w, cell_h))
    cv2.putText(v_raw, "1. RAW", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # --- 2. GT JSON (우상) ---
    v_gt = img.copy()
    json_path = Path(paths['interp_data']) / f"{frame_name}.json" 
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            if data['instance_info']:
                inst = data['instance_info'][0]
                gt_kpts = np.array([[k[0], k[1]] for k in inst['keypoints']])
                v_gt = draw_skeleton_body(v_gt, gt_kpts, color=(0, 255, 0))
    v_gt = cv2.resize(v_gt, (cell_w, cell_h))
    cv2.putText(v_gt, "2. GT (Body Only)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # --- 3. Single-Stage (좌하) ---
    v_s = img.copy()
    p_single = vid_preds.get(frame_name, {}).get('single', None)
    if p_single is not None:
        v_s = draw_skeleton_body(v_s, p_single, color=(0, 0, 255))
    v_s = cv2.resize(v_s, (cell_w, cell_h))
    cv2.putText(v_s, "3. SINGLE-STAGE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # --- 4. Double-Stage (우하) ---
    v_d = img.copy()
    p_double = vid_preds.get(frame_name, {}).get('double', None)
    if p_double is not None:
        v_d = draw_skeleton_body(v_d, p_double, color=(0, 255, 255))
    v_d = cv2.resize(v_d, (cell_w, cell_h))
    cv2.putText(v_d, "4. DOUBLE-STAGE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # 병합
    combined = np.vstack([np.hstack([v_raw, v_gt]), np.hstack([v_s, v_d])])
    video_out.write(combined)

video_out.release()
print(f"🎉 영상 저장 완료: {out_name}")