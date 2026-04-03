import sys
import shutil
import pickle
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 💡 OpenCV와 PyTorch의 스레드 충돌 방지
cv2.setNumThreads(0)

# =================================================================
# 0. 유틸리티 함수 (Letterbox 및 좌표 복구)
# =================================================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """원본 비율을 유지하며 패딩을 추가하여 리사이즈 (YOLO 내부 방식 모방)"""
    shape = im.shape[:2] 
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def scale_coords_back(kpts, ratio, pad, offset):
    """Crop 이미지 내 좌표를 원본 이미지 좌표계로 역산"""
    kpts[:, 0] -= pad[0]
    kpts[:, 1] -= pad[1]
    kpts[:, :2] /= ratio
    kpts[:, 0] += offset[0]  # x1
    kpts[:, 1] += offset[1]  # y1
    return kpts

# =================================================================
# 1. 경로 설정 및 데이터 추출 (사용자 로직 포함)
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))
from utils.path_list import path_list

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA1_PATH = DATA_DIR / "metadata_v2.0.csv"
METADATA2_PATH = DATA_DIR / "metadata_v2.1.csv"
CACHE_DIR = Path("/tmp/yolo_pose_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_changed_data(p1, p2):
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    merged = pd.merge(df2, df1[['common_path', 'is_train', 'is_val']], on='common_path', how='left', suffixes=('', '_old'))
    condition = (((merged['is_train'] == True) & (merged['is_train_old'] != True)) | 
                 ((merged['is_val'] == True) & (merged['is_val_old'] != True)))
    return merged[condition].reset_index(drop=True)

target_df = get_changed_data(METADATA1_PATH, METADATA2_PATH)
print(f"✅ 분석 대상 비디오 개수: {len(target_df)}개")

# =================================================================
# 2. 데이터 로컬 캐싱 (NAS 병목 제거)
# =================================================================
LIST_CACHE_FILE = CACHE_DIR / "all_items_list.pkl"
if LIST_CACHE_FILE.exists():
    print("\n🚀 [로컬 데이터 캐시 로드 중...]")
    with open(LIST_CACHE_FILE, 'rb') as f:
        all_items = pickle.load(f)
else:
    print("\n🚀 [최초 1회 데이터 스캔 및 로컬 캐싱 중...]")
    all_items = []
    for cp in tqdm(target_df['common_path'], desc="NAS -> 로컬 복사"):
        paths_dict = path_list(cp)
        frame_dir = Path(paths_dict['frame'])
        txt_dir_yolo = Path(paths_dict.get('yolo_txt', ''))
        txt_dir_interp = Path(paths_dict['interp_data'])
        
        for f_path in frame_dir.glob("*.jpg"):
            txt_path = txt_dir_yolo / f"{f_path.stem}.txt"
            if not txt_path.exists():
                txt_path = txt_dir_interp / f"{f_path.stem}.txt"
            
            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    gt_line = f.readline().strip()
                if not gt_line: continue
                
                local_img_path = CACHE_DIR / f"{str(cp).replace('/', '_')}_{f_path.name}"
                if not local_img_path.exists():
                    shutil.copy2(f_path, local_img_path)
                all_items.append((str(local_img_path), gt_line))

    with open(LIST_CACHE_FILE, 'wb') as f:
        pickle.dump(all_items, f)

# =================================================================
# 3. 모델 로드 및 비교 평가
# =================================================================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SIGMAS = torch.tensor([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 
                       0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089], device=device)

def calculate_oks_torch(pred_kpts, gt_kpts, scale, sigmas):
    dists_sq = torch.sum((pred_kpts[:, :2] - gt_kpts[:, :2])**2, dim=1)
    v_mask = gt_kpts[:, 2] > 0 
    if torch.sum(v_mask) == 0: return 0.0
    denom = 2 * (scale**2) * (sigmas**2)
    return (torch.sum(torch.exp(-dists_sq / denom)[v_mask]) / torch.sum(v_mask)).item()

print(f"\n🚀 모델 로드 및 평가 시작...")
pose_model = YOLO(DATA_DIR / "checkpoints/YOLO/yolo11n-pose.pt").to(device)
det_model = YOLO('yolo11.pt').to(device) # Person detector

metrics = {
    'Single-Stage': {'okss': [], 'det': 0},
    'Two-Stage(Crop)': {'okss': [], 'det': 0}
}

BATCH_SIZE = 32
executor = ThreadPoolExecutor(max_workers=8)

for i in tqdm(range(0, len(all_items), BATCH_SIZE), desc="배치 처리"):
    batch = all_items[i:i+BATCH_SIZE]
    imgs = list(executor.map(cv2.imread, [b[0] for b in batch]))
    gts = [b[1] for b in batch]

    # Method 1
    res_s = pose_model.predict(imgs, verbose=False, device=device)
    # Method 2
    res_d = det_model.predict(imgs, classes=[0], verbose=False, device=device)

    for idx, (r_s, r_d, gt_line) in enumerate(zip(res_s, res_d, gts)):
        img_o = imgs[idx]
        h_o, w_o = img_o.shape[:2]
        
        # GT 파싱
        p = list(map(float, gt_line.split()))
        scale = np.sqrt((p[3]*w_o)*(p[4]*h_o))
        gt_t = torch.tensor(np.array([[p[j]*w_o, p[j+1]*h_o, p[j+2]] for j in range(5, len(p), 3)]), device=device, dtype=torch.float32)

        # 1. Single-Stage
        if r_s.keypoints is not None and len(r_s.keypoints.xy) > 0:
            metrics['Single-Stage']['okss'].append(calculate_oks_torch(r_s.keypoints.xy[0][5:17], gt_t, scale, SIGMAS[5:17]))
            metrics['Single-Stage']['det'] += 1

        # 2. Two-Stage
        if len(r_d.boxes) > 0:
            b = r_d.boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = b
            crop = img_o[max(0, y1):min(h_o, y2), max(0, x1):min(w_o, x2)]
            if crop.size > 0:
                input_p, ratio, pad = letterbox(crop, new_shape=640)
                r_p = pose_model.predict(input_p, verbose=False, device=device)[0]
                if r_p.keypoints is not None and len(r_p.keypoints.xy) > 0:
                    pred_p = scale_coords_back(r_p.keypoints.xy[0].clone(), ratio, pad, (x1, y1))
                    metrics['Two-Stage(Crop)']['okss'].append(calculate_oks_torch(pred_p[5:17], gt_t, scale, SIGMAS[5:17]))
                    metrics['Two-Stage(Crop)']['det'] += 1

# =================================================================
# 4. 결과 리포트
# =================================================================
report = []
for name, m in metrics.items():
    report.append({
        "Strategy": name,
        "Det Rate (%)": round((m['det'] / len(all_items)) * 100, 2),
        "Mean OKS": round(np.mean(m['okss']), 4) if m['okss'] else 0
    })

print("\n" + "="*60)
print(pd.DataFrame(report).to_string(index=False))
print("="*60)