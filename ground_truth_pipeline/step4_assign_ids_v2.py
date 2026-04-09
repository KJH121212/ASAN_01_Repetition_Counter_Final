import cv2 
import json 
import math
import random 
import numpy as np 
from pathlib import Path 
from tqdm import tqdm 

# 관절별 중요도 가중치 정의
KPT_WEIGHTS = {
    0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, # 얼굴 (눈코입귀)
    5: 2.0, 6: 2.0,                         # 어깨 (중요)
    7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0,        # 팔 (엘보, 손목)
    11: 2.0, 12: 2.0,                       # 골반 (중요)
    13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0      # 다리 (무릎, 발목)
}

def rle_to_mask(segmentation_dict): 
    counts = segmentation_dict.get('counts') 
    size = segmentation_dict.get('size') 
    if not counts or not size: return None 
    
    h, w = size[0], size[1] 
    mask = np.zeros(h * w, dtype=np.uint8) 
    
    rle = np.array(counts) 
    starts = rle[0::2] - 1 
    lengths = rle[1::2] 
    ends = starts + lengths 
    
    for lo, hi in zip(starts, ends): 
        mask[max(lo, 0):min(hi, len(mask))] = 1 
        
    return mask.reshape((h, w)) 

def get_bbox_from_mask(mask): 
    coords = np.argwhere(mask > 0) 
    if coords.size == 0: return None 
    
    y_min, x_min = coords.min(axis=0) 
    y_max, x_max = coords.max(axis=0) 
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)] 

def calculate_iou(box1, box2): 
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1]) 
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3]) 
    if x_right < x_left or y_bottom < y_top: return 0.0 
    inter_area = (x_right - x_left) * (y_bottom - y_top) 
    area1, area2 = (box1[2]-box1[0])*(box1[3]-box1[1]), (box2[2]-box2[0])*(box2[3]-box2[1]) 
    if float(area1 + area2 - inter_area) == 0: return 0.0
    return inter_area / float(area1 + area2 - inter_area) 

def filter_duplicate_skeletons(instance_info, iou_threshold=0.65):
    """
    중복 탐지된 스켈레톤을 IoU(Intersection over Union) 기준으로 필터링하여 
    가장 점수가 높은 인스턴스만 남기는 함수 (NMS 로직)
    """
    # 입력된 인스턴스 정보가 없으면 빈 리스트 반환
    if not instance_info: return []

    # 1. 각 인스턴스의 신뢰도 합산 (우선순위 결정 기준)
    for inst in instance_info:
        # 17개 관절의 keypoint_scores를 모두 더해 해당 스켈레톤의 전체 점수 계산
        inst['total_score'] = sum(inst.get('keypoint_scores', []))
    
    # 2. 전체 점수(total_score)를 기준으로 내림차순 정렬
    # 점수가 높은(더 정확해 보이는) 스켈레톤이 리스트의 앞쪽으로 오게 함
    instance_info.sort(key=lambda x: x['total_score'], reverse=True)
    
    filtered_info = [] # 최종적으로 살아남은 인스턴스들을 담을 리스트
    
    # 3. 중복 제거 루프 (NMS 핵심 로직)
    for inst in instance_info:
        # 현재 검사할 인스턴스의 Bbox 좌표 추출 및 평탄화
        raw_box = np.array(inst.get('bbox', [])).flatten().tolist()
        # [x1, y1, x2, y2] 형태의 4개 좌표만 슬라이싱 (없으면 [0,0,0,0])
        box_a = raw_box[:4] if len(raw_box) >= 4 else [0,0,0,0]
        
        keep = True # 보관 여부 플래그
        
        # 이미 필터링을 통과해 보관된 인스턴스들과 비교
        for keep_inst in filtered_info:
            raw_box_k = np.array(keep_inst.get('bbox', [])).flatten().tolist()
            box_b = raw_box_k[:4] if len(raw_box_k) >= 4 else [0,0,0,0]
            
            # 두 Bbox의 겹침 정도(IoU)가 임계값(85%)을 넘으면 동일 객체로 판단
            if calculate_iou(box_a, box_b) > iou_threshold:
                # 이미 점수가 더 높은 객체(keep_inst)가 있으므로 현재 객체는 버림
                keep = False 
                break
                
        # 어떤 기존 객체와도 많이 겹치지 않는다면(새로운 인물이면) 추가
        if keep:
            # 계산용으로 썼던 임시 키 'total_score'는 제거 후 저장
            inst.pop('total_score', None)
            filtered_info.append(inst)
            
    return filtered_info

# =================================================================
# 🌟 [메인 핵심 함수] 가중치 기반 정밀 픽셀 매칭 및 탐욕적 할당 적용
# =================================================================
# =================================================================
# 🌟 [메인 핵심 함수] 가중치 기반 정밀 픽셀 매칭 및 탐욕적 할당 적용
# =================================================================
def assign_sam_ids_to_keypoints(sam_dir, kpt_dir, output_dir=None): 
    save_dir = Path(output_dir) if output_dir else Path(kpt_dir) 
    save_dir.mkdir(parents=True, exist_ok=True) 
    processed_count = 0 
    
    # 💡 [수정 포인트 1]: glob 대신 rglob을 사용하여 하위 폴더(01, 02...)의 모든 json을 탐색합니다.
    all_kpt_files = sorted(Path(kpt_dir).rglob("*.json"))

    for kpt_path in tqdm(all_kpt_files, desc="Aligning Skeletons & Masks"):
        # 💡 [수정 포인트 2]: 최상위 폴더(kpt_dir)를 제외한 하위 폴더 경로(예: '01/000000.json')를 추출합니다.
        rel_path = kpt_path.relative_to(Path(kpt_dir))
        
        # 💡 [수정 포인트 3]: sam_dir 내에서도 동일한 하위 폴더 구조를 갖도록 경로를 생성합니다.
        # with_name을 사용하여 부모 경로('01/')는 유지하되 파일명만 'sam_...'으로 변경해 확인합니다.
        sam_path = Path(sam_dir) / rel_path.with_name(f"sam_{rel_path.name}") 
        if not sam_path.exists(): 
            sam_path = Path(sam_dir) / rel_path 
        
        if not sam_path.exists(): continue 

        try:
            with open(kpt_path, 'r', encoding='utf-8') as f: s_data = json.load(f) 
            with open(sam_path, 'r', encoding='utf-8') as f: m_data = json.load(f) 
        except Exception as e:
            continue
        
        # -------------------------------------------------------------------
        # [데이터 처리 및 매칭 로직은 경로와 무관하므로 기존과 100% 동일합니다]
        # -------------------------------------------------------------------
        original_instances = s_data.get('instance_info', [])
        instances = filter_duplicate_skeletons(original_instances)

        sam_info_list = [] 
        for obj in m_data.get('objects', []): 
            mask_2d = rle_to_mask(obj.get('segmentation', {})) 
            if mask_2d is not None: 
                exact_bbox = get_bbox_from_mask(mask_2d)
                if exact_bbox:
                    x1, y1, x2, y2 = exact_bbox
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    sam_info_list.append({
                        'id': obj['id'], 
                        'mask': mask_2d, 
                        'bbox': exact_bbox, 
                        'center': (cx, cy)
                    })

        match_scores = [] 

        for inst_idx, inst in enumerate(instances):
            inst['instance_id'] = None 
            
            kpts = inst.get('keypoints', [])
            kpt_scores = inst.get('keypoint_scores', [])
            
            valid_points = []
            total_possible_weight = 0.0 
            
            for k_idx, (kx, ky) in enumerate(kpts):
                conf = kpt_scores[k_idx] if isinstance(kpt_scores, list) and k_idx < len(kpt_scores) else 0.0
                if isinstance(conf, list): conf = conf[0]
                
                if conf > 0.008 and kx > 0 and ky > 0:
                    valid_points.append([kx, ky, k_idx, conf])
                    total_possible_weight += KPT_WEIGHTS.get(k_idx, 1.0) * conf
            
            if valid_points:
                pts_array = np.array([[p[0], p[1]] for p in valid_points])
                min_x, min_y = np.min(pts_array, axis=0)
                max_x, max_y = np.max(pts_array, axis=0)
                pad = 15
                sx1, sy1 = int(max(0, min_x - pad)), int(max(0, min_y - pad))
                sx2, sy2 = int(max_x + pad), int(max_y + pad)
            else:
                sx1, sy1, sx2, sy2 = 0, 0, 0, 0
            
            inst['bbox'] = [sx1, sy1, sx2, sy2]
            skel_cx, skel_cy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0

            for m_idx, m_info in enumerate(sam_info_list):
                mask_2d = m_info['mask']
                m_bbox = m_info['bbox']
                mh, mw = mask_2d.shape
                
                weighted_hit_score = 0.0 
                
                for kx, ky, k_idx, conf in valid_points:
                    weight = KPT_WEIGHTS.get(k_idx, 1.0)
                    ix, iy = int(kx), int(ky)
                    
                    if (0 <= ix < mw) and (0 <= iy < mh) and (mask_2d[iy, ix] == 1):
                        weighted_hit_score += (weight * conf * 1.0)
                    elif (m_bbox[0] <= ix <= m_bbox[2]) and (m_bbox[1] <= iy <= m_bbox[3]):
                        weighted_hit_score += (weight * conf * 0.3)

                ratio = (weighted_hit_score / total_possible_weight) if total_possible_weight > 0 else 0.0
                iou = calculate_iou([sx1, sy1, sx2, sy2], m_bbox)
                dist = math.hypot(skel_cx - m_info['center'][0], skel_cy - m_info['center'][1])
                
                if ratio >= 0.6 or (ratio >= 0.4 and iou >= 0.2):
                    match_scores.append((ratio, iou, -dist, inst_idx, m_idx))

        match_scores.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        
        assigned_insts = set() 
        assigned_masks = set() 

        for ratio, iou, inv_dist, inst_idx, m_idx in match_scores:
            if inst_idx not in assigned_insts and m_idx not in assigned_masks:
                chosen = sam_info_list[m_idx]
                instances[inst_idx]['instance_id'] = int(chosen['id'])
                instances[inst_idx]['bbox'] = chosen['bbox']
                
                assigned_insts.add(inst_idx)
                assigned_masks.add(m_idx)
                
            if len(assigned_insts) == len(instances) or len(assigned_masks) == len(sam_info_list):
                break

        # 5. 결과 저장
        s_data['instance_info'] = instances 
        
        # 💡 [수정 포인트 4]: 저장할 때도 원본과 동일한 하위 폴더 구조를 따라갑니다.
        final_save_path = save_dir / rel_path # 예: 4_INTERP_DATA/subject/action/01/000000.json
        final_save_path.parent.mkdir(parents=True, exist_ok=True) # 폴더가 없을 경우를 대비해 생성합니다.
        
        with open(final_save_path, 'w', encoding='utf-8') as f:
            json.dump(s_data, f, indent=4)
        processed_count += 1
            
    return processed_count