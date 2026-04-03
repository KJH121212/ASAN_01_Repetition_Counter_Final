import sys # 시스템 설정을 위한 모듈입니다.
import torch # GPU 할당 및 DataLoader 사용을 위한 파이토치 라이브러리입니다.
from torch.utils.data import Dataset, DataLoader # 💡 NAS 병목을 해결할 핵심 무기입니다!
import numpy as np # 수학적 연산(mAP, 거리 계산 등)을 위한 넘파이입니다.
import pandas as pd # 메타데이터 CSV를 제어하기 위한 판다스입니다.
import cv2 # 이미지 로드를 위한 OpenCV입니다.
import matplotlib.pyplot as plt # 결과 표를 이미지로 저장하기 위한 모듈입니다.
from pathlib import Path # 안전한 경로 처리를 위한 Path 객체입니다.
from ultralytics import YOLO # YOLO 모델을 불러오기 위한 모듈입니다.
from tqdm import tqdm # 진행률을 예쁘게 보여주는 프로그레스 바입니다.

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))
from utils.path_list import path_list

# =================================================================
# 0. 평가지표 산출을 위한 수학 함수 및 COCO 상수 정의
# =================================================================
SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
])

def calculate_oks(pred_kpts, gt_kpts, scale, sigmas):
    dists_sq = np.sum((pred_kpts[:, :2] - gt_kpts[:, :2])**2, axis=1)
    v_mask = gt_kpts[:, 2] > 0 
    if np.sum(v_mask) == 0: return 0.0
    denom = 2 * (scale**2) * (sigmas**2)
    oks = np.exp(-dists_sq / denom)
    return np.sum(oks[v_mask]) / np.sum(v_mask)

def calculate_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# =================================================================
# 1. 경로 설정 및 대상 데이터 추출
# =================================================================
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA1_PATH = DATA_DIR / "metadata_v2.0.csv"
METADATA2_PATH = DATA_DIR / "metadata_v2.1.csv"

paths_to_check = [
    DATA_DIR / "checkpoints/YOLO/yolo11n-pose.pt",
    DATA_DIR / "checkpoints/YOLO_FINETUNING/v1.0_step1/weights/best.pt",
    DATA_DIR / "checkpoints/YOLO_FINETUNING/v1.0_step15/weights/best.pt",
    DATA_DIR / "checkpoints/YOLO_FINETUNING/v1.0_step30/weights/best.pt",
    DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step1/weights/best.pt",
    DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step10/weights/best.pt",
    DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step30/weights/best.pt"
]

def get_changed_data(p1, p2):
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    merged = pd.merge(df2, df1[['common_path', 'is_train', 'is_val']], on='common_path', how='left', suffixes=('', '_old'))
    condition = (((merged['is_train'] == True) & (merged['is_train_old'] != True)) | 
                 ((merged['is_val'] == True) & (merged['is_val_old'] != True)))
    return merged[condition].reset_index(drop=True)

target_df = get_changed_data(METADATA1_PATH, METADATA2_PATH)
print(f"✅ 분석 대상 비디오 개수: {len(target_df)}개")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"🚀 연산 디바이스 설정 완료: {device.upper()} 모드로 구동됩니다!")

# =================================================================
# 2. 💡 NAS 데이터 병목 돌파를 위한 커스텀 Dataset 정의
# =================================================================
class NASPoseDataset(Dataset):
    def __init__(self, item_list):
        self.item_list = item_list # (img_path, txt_path) 튜플 리스트를 받습니다.

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        # 💡 백그라운드 워커들이 병렬로 이 함수를 실행하여 이미지를 퍼옵니다!
        img_path, txt_path = self.item_list[idx]
        
        img = cv2.imread(str(img_path))
        with open(txt_path, 'r') as f:
            gt_line = f.readline().strip()
            
        h, w = img.shape[:2]
        return img, gt_line, h, w

def custom_collate(batch):
    # PyTorch의 기본 Tensor 변환을 막고, YOLO가 좋아하는 List 형태로 예쁘게 묶어줍니다.
    imgs, gts, hs, ws = zip(*batch)
    return list(imgs), list(gts), list(hs), list(ws)

import cv2
cv2.setNumThreads(0) # 💡 OpenCV와 PyTorch의 스레드 충돌을 막아 CPU 병목을 해소하는 마법의 옵션입니다.

# =================================================================
# 3. 모델 로드 및 초고속 다이렉트 평가 (메모리 공유 최적화 🚀)
# =================================================================
print("\n🚀 [2. 모델 7개 GPU 로드 중...]")
loaded_models = {}
for m_path in paths_to_check:
    if not m_path.exists(): continue
    m_name = m_path.parts[-3] if "weights" in str(m_path) else "Pretrained"
    model = YOLO(m_path)
    model.to(device)
    loaded_models[m_name] = model

metrics_tracker = {
    name: {'total_frames': 0, 'detected': 0, 'pck_corr': 0, 'pck_tot': 0, 'pix_errs': [], 'stats': []}
    for name in loaded_models.keys()
}

BATCH_SIZE = 128 
print(f"\n🚀 [3. 초고속 다이렉트 평가 시작 (Batch Size: {BATCH_SIZE})]")

for i in tqdm(range(0, len(all_items), BATCH_SIZE), desc="배치 처리 진행률"):
    batch_items = all_items[i:i+BATCH_SIZE]
    batch_paths = [item[0] for item in batch_items] 
    batch_gts = [item[1] for item in batch_items]   
    
    # 💡 [핵심 최적화] 7번 읽을 것을 딱 1번만 읽어서 RAM에 올립니다!
    # 여기서 읽은 이미지 배열(Numpy) 리스트를 7개의 모델이 공유합니다.
    batch_imgs = [cv2.imread(p) for p in batch_paths]
    
    for m_name, model in loaded_models.items():
        # 💡 경로(문자열)가 아닌 '메모리에 올라간 이미지 배열'을 던져줍니다. (디스크 I/O 0%)
        results = model.predict(source=batch_imgs, batch=BATCH_SIZE, stream=True, verbose=False, device=device)
        
        tracker = metrics_tracker[m_name]
        
        for res, gt_line in zip(results, batch_gts):
            h, w = res.orig_shape 
            
            parts = list(map(float, gt_line.split()))
            bw_px, bh_px = parts[3] * w, parts[4] * h
            scale = np.sqrt(bw_px * bh_px)  
            diag = np.sqrt(bw_px**2 + bh_px**2) 
            
            gt_kpts = np.array([[parts[i]*w, parts[i+1]*h, parts[i+2]] for i in range(5, len(parts), 3)])
            tracker['total_frames'] += 1
            
            if res.keypoints is not None and len(res.keypoints) > 0 and hasattr(res.keypoints, 'xy') and len(res.keypoints.xy) > 0:
                tracker['detected'] += 1
                pred_kpts = res.keypoints.xy[0].cpu().numpy()
                conf_score = res.boxes.conf[0].cpu().item() 
                
                target_sigmas = SIGMAS[5:17] 
                if len(pred_kpts) == 17 and len(gt_kpts) == 12:
                    p_kpts, g_kpts = pred_kpts[5:17], gt_kpts
                else:
                    min_len = min(len(pred_kpts), len(gt_kpts))
                    p_kpts, g_kpts, target_sigmas = pred_kpts[:min_len], gt_kpts[:min_len], target_sigmas[:min_len]
                
                current_oks = calculate_oks(p_kpts, g_kpts, scale, target_sigmas)
                tracker['stats'].append((conf_score, current_oks))
                
                v_mask = g_kpts[:, 2] > 0
                dists = np.sqrt(np.sum((p_kpts[:, :2] - g_kpts[:, :2])**2, axis=1))
                
                tracker['pix_errs'].extend(dists[v_mask])
                tracker['pck_tot'] += np.sum(v_mask)
                tracker['pck_corr'] += np.sum(dists[v_mask] < (diag * 0.1))
            else:
                tracker['stats'].append((0.0, 0.0))

# =================================================================
# 4. 데이터셋 전체를 대상으로 mAP 연산 및 지표 저장
# =================================================================
results_list = []
for m_name, tracker in metrics_tracker.items():
    t_frames = tracker['total_frames']
    stats = sorted(tracker['stats'], key=lambda x: x[0], reverse=True)
    confs, okss = zip(*stats) if stats else ([], [])
    okss = np.array(okss)
    
    ap_list = []
    thresholds = np.linspace(0.5, 0.95, 10) 
    
    for thr in thresholds:
        tp = np.cumsum(okss >= thr)
        fp = np.cumsum(okss < thr)
        recalls = tp / t_frames if t_frames > 0 else np.array([0])
        precisions = tp / (tp + fp + 1e-16)
        ap_list.append(calculate_ap(recalls, precisions))

    results_list.append({
        "Model": m_name,
        "1. Det_Rate(%)": round((tracker['detected'] / t_frames * 100), 2) if t_frames else 0,
        "2. mAP@.5:.95": round(np.mean(ap_list), 4) if ap_list else 0.0,
        "3. AP50 / 75": f"{round(ap_list[0], 4):.4f} / {round(ap_list[5], 4):.4f}" if ap_list else "0.0000 / 0.0000",
        "4. PCK@0.1(%)": round((tracker['pck_corr'] / tracker['pck_tot'] * 100), 2) if tracker['pck_tot'] else 0,
        "5. Avg_Pix_Err": round(np.mean(tracker['pix_errs']), 2) if tracker['pix_errs'] else 0.0
    })

# =================================================================
# 5. 결과 표 출력 및 시각화(PNG)
# =================================================================
report_df = pd.DataFrame(results_list)

# 💡 [콘솔 출력용] 간격이 완벽하게 맞는 커스텀 표 그리기
print("\n" + "="*98)
print(f"📊 [포즈 추정 모델 최종 종합 평가 리포트]")
print("="*98)
# 헤더(컬럼명)의 간격을 명시적으로 고정합니다 (<: 왼쪽 정렬, >: 오른쪽 정렬)
header = f"{'Model':<15} | {'1. Det_Rate(%)':>14} | {'2. mAP@.5:.95':>13} | {'3. AP50 / 75':>17} | {'4. PCK@0.1(%)':>13} | {'5. Avg_Pix_Err':>14}"
print(header)
print("-" * 98)

# 데이터 행들을 순회하며 소수점과 간격을 강제로 맞춰서 출력합니다.
for _, row in report_df.iterrows():
    model_name = str(row['Model'])
    det = f"{float(row['1. Det_Rate(%)']):.2f}"
    map_val = f"{float(row['2. mAP@.5:.95']):.4f}"
    ap_str = str(row['3. AP50 / 75'])
    pck = f"{float(row['4. PCK@0.1(%)']):.2f}"
    err = f"{float(row['5. Avg_Pix_Err']):.2f}"
    
    # 헤더와 동일한 간격(15, 14, 13, 17, 13, 14)을 부여하여 세로줄(|)이 완벽히 일치하게 만듭니다.
    print(f"{model_name:<15} | {det:>14} | {map_val:>13} | {ap_str:>17} | {pck:>13} | {err:>14}")
print("="*98 + "\n")

# # 💡 [이미지 저장용] 표 렌더링
# fig, ax = plt.subplots(figsize=(12, 3)) 
# ax.axis('tight'); ax.axis('off')
# tbl = ax.table(cellText=report_df.values, colLabels=report_df.columns, loc='center', cellLoc='center')
# tbl.auto_set_font_size(False); tbl.set_fontsize(11)
# plt.savefig("./compare_result_final.png", bbox_inches='tight')