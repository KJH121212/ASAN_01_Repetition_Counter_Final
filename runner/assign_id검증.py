import sys 
import cv2 
import json 
import math 
import numpy as np 
import pandas as pd 
from pathlib import Path 

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR)) 

from utils.path_list import path_list 

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"

# =================================================================
# 2. 임시 헬퍼 함수
# =================================================================
def _calc_iou(boxA, boxB): 
    xA = max(boxA[0], boxB[0]) 
    yA = max(boxA[1], boxB[1]) 
    xB = min(boxA[2], boxB[2]) 
    yB = min(boxA[3], boxB[3]) 
    interArea = max(0, xB - xA) * max(0, yB - yA) 
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) 
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) 
    if float(boxAArea + boxBArea - interArea) == 0: return 0.0 
    return interArea / float(boxAArea + boxBArea - interArea) 

def rle_to_mask(segmentation_dict): # RLE 마스크 복원 함수
    counts = segmentation_dict.get('counts') # 압축 데이터
    size = segmentation_dict.get('size') # 원본 이미지 크기
    if not counts or not size: return None # 누락 시 종료
    
    h, w = size[0], size[1] # 높이, 너비 추출
    mask = np.zeros(h * w, dtype=np.uint8) # 1차원 빈 마스크 생성
    
    rle = np.array(counts) # numpy 배열로 변환
    starts = rle[0::2] - 1 # 시작 픽셀 (0-based)
    lengths = rle[1::2] # 칠할 길이
    ends = starts + lengths # 끝 픽셀
    
    for lo, hi in zip(starts, ends): # 각 구간 반복
        mask[max(lo, 0):min(hi, len(mask))] = 1 # 마스크 영역 1로 채우기
        
    return mask.reshape((h, w)) # 2차원으로 접어 반환



# =================================================================
# 3. 데이터 로드 및 특정 프레임 설정
# =================================================================
print(f"📂 CSV 로드 중... ({METADATA_PATH.name})") 
meta_df = pd.read_csv(METADATA_PATH) 

target = 1009 
common_path = meta_df.iloc[target]['common_path'] 
paths = path_list(common_path) 

target_idx = 42365 
frame_name = f"{target_idx:06d}" 

img_path = Path(paths['frame']) / f"{frame_name}.jpg" 
kpt_path = Path(paths['keypoint']) / f"{frame_name}.json" 
sam_path = Path(paths['sam']) / f"{frame_name}.json" 

out_dir = Path(paths['test']) if 'test' in paths else Path("./test_output") 
out_dir.mkdir(parents=True, exist_ok=True) 

if not (img_path.exists() and kpt_path.exists() and sam_path.exists()): 
    print(f"❌ [Error] {frame_name} 프레임에 필요한 파일이 누락되었습니다.") 
    sys.exit() 

img = cv2.imread(str(img_path)) 
img_h, img_w = img.shape[:2] 

with open(kpt_path, 'r', encoding='utf-8') as f: s_data = json.load(f) 
with open(sam_path, 'r', encoding='utf-8') as f: m_data = json.load(f) 

# =================================================================
# 4. 검증 로직 및 시각화 (RLE Pixel 단위 + Ratio 방어선)
# =================================================================
instances = s_data.get('instance_info', []) 
sam_objects = m_data.get('objects', []) 

debug_img = (img * 0.4).astype(np.uint8) # 텍스트 가독성을 위해 배경을 어둡게

print(f"\n🔍 [Frame {frame_name} 분석 결과 - PIXEL MATCHING]") 

for m_idx, obj in enumerate(sam_objects): 
    # 🌟 1. RLE 디코딩하여 2D 픽셀 마스크 획득
    seg_data = obj.get('segmentation', {})
    mask_2d = rle_to_mask(seg_data) 
    if mask_2d is None: continue
    
    mh, mw = mask_2d.shape # 마스크의 해상도
    
    # 마스크에서 정확한 Bbox 추출 (IoU 및 거리 계산용)
    exact_bbox = get_bbox_from_mask(mask_2d)
    if not exact_bbox: continue
    x1, y1, x2, y2 = exact_bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 
    
    # 🌟 2. 마스크 영역 시각화 (반투명 파란색으로 픽셀을 직접 칠해줍니다)
    colored_mask = np.zeros_like(debug_img)
    colored_mask[mask_2d == 1] = [255, 0, 0] # BGR의 Blue
    cv2.addWeighted(debug_img, 1.0, colored_mask, 0.3, 0, debug_img) # 30% 투명도로 오버레이
    
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1) # Bbox는 얇게 참고용으로만 그림
    cv2.putText(debug_img, f"Mask {m_idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) 

    for inst_idx, inst in enumerate(instances): 
        kpts = inst.get('keypoints', []) 
        kpt_scores = inst.get('keypoint_scores', [])
        
        # 유효 관절 파악 및 Bbox 실시간 복원
        valid_points = []
        total_valid_pts = 0.0 
        
        for k_idx, (kx, ky) in enumerate(kpts):
            conf = kpt_scores[k_idx] if isinstance(kpt_scores, list) and k_idx < len(kpt_scores) else 0.0
            if isinstance(conf, list): conf = conf[0]
            if conf > 0.05 and kx > 0 and ky > 0:
                valid_points.append([kx, ky])
                total_valid_pts += 1.0 
                
        if valid_points:
            pts_array = np.array(valid_points)
            min_x, min_y = np.min(pts_array, axis=0)
            max_x, max_y = np.max(pts_array, axis=0)
            pad = 15
            sx1, sy1 = int(max(0, min_x - pad)), int(max(0, min_y - pad))
            sx2, sy2 = int(min(img_w, max_x + pad)), int(min(img_h, max_y + pad))
        else:
            sx1, sy1, sx2, sy2 = 0, 0, 0, 0
            
        skel_cx, skel_cy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0 
        cv2.rectangle(debug_img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 1) 
        
        # 🌟 3. Bbox가 아닌 실제 픽셀(mask_2d) 단위 Hit 판별!
        hit_votes = 0.0 
        for k_idx, (kx, ky) in enumerate(kpts): 
            conf = kpt_scores[k_idx] if isinstance(kpt_scores, list) and k_idx < len(kpt_scores) else 0.0
            if isinstance(conf, list): conf = conf[0]
            if conf <= 0.05: continue 

            ix, iy = int(kx), int(ky) 
            
            # ix, iy가 해상도를 벗어나지 않고, 해당 픽셀의 마스크 값이 1(내부)일 때만 Hit!
            if (0 <= ix < mw) and (0 <= iy < mh) and (mask_2d[iy, ix] == 1): 
                cv2.circle(debug_img, (ix, iy), 4, (0, 0, 255), -1) # Hit (빨강)
                hit_votes += 1.0 
            else: 
                cv2.circle(debug_img, (ix, iy), 2, (0, 255, 255), -1) # Miss (노랑)

        # 채점 및 방어선 로직
        iou = _calc_iou([sx1, sy1, sx2, sy2], [x1, y1, x2, y2]) 
        dist = math.hypot(skel_cx - cx, skel_cy - cy) 
        
        ratio = (hit_votes / total_valid_pts) if total_valid_pts > 0 else 0.0
        
        # 컷오프 조건
        is_pass = ratio >= 0.6 or (ratio >= 0.4 and iou >= 0.2)
        
        pass_str = "PASS" if is_pass else "FAIL"
        color = (0, 255, 0) if is_pass else (150, 150, 150)
        
        score_text = f"Skel {inst_idx} vs Mask {m_idx} | Ratio:{ratio:.2f} ({int(hit_votes)}/{int(total_valid_pts)}) | IoU:{iou:.2f} -> [{pass_str}]" 
        
        text_y = 50 + (m_idx * len(instances) + inst_idx) * 30 
        cv2.putText(debug_img, score_text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) 
        print(f"   -> {score_text}") 

save_path = out_dir / f"score_debug_frame_{frame_name}.jpg" 
cv2.imwrite(str(save_path), debug_img) 
print(f"\n✅ 픽셀 단위 검증용 이미지가 저장되었습니다: {save_path}")