import json
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # 🟢 tqdm 임포트

# =================================================================
# 🛠️ [Helper] 유틸리티 함수 (기존과 동일)
# =================================================================
def get_box_details(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        raise ValueError(f"Invalid bbox format: {bbox}")
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return (cx, cy), area

def calculate_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

# =================================================================
# 🚀 [Core] ID 할당 로직
# =================================================================

def assign_sam_ids_to_keypoints(
    common_path: str, 
    sam_dir: Path,       # SAM 폴더 경로 직접 입력
    kpt_dir: Path,       # Keypoint 폴더 경로 직접 입력
    output_base_dir: Path = None
):
    """
    특정 시퀀스에 대해 SAM 결과와 Keypoint 결과를 매칭하여 ID를 할당합니다.
    """
    
    # 1. 출력 경로 설정
    if output_base_dir is None:
        save_dir = kpt_dir # 덮어쓰기
    else:
        save_dir = output_base_dir / common_path
        save_dir.mkdir(parents=True, exist_ok=True) # 폴더 생성

    # 2. 폴더 존재 확인
    if not kpt_dir.exists():
        return 0

    kpt_files = sorted(list(kpt_dir.glob("*.json")))
    if not kpt_files:
        return 0

    processed_count = 0
    
    # 3. 파일 순회 및 매칭 로직 (🟢 tqdm 추가)
    # leave=False: 완료 시 진행바를 지워서 터미널을 깔끔하게 유지
    for kpt_path in tqdm(kpt_files, desc="   Matching IDs", leave=False, unit="file"):
        
        # 대응되는 SAM 파일 찾기
        sam_path = sam_dir / f"sam_{kpt_path.name}"
        if not sam_path.exists():
            sam_path = sam_dir / kpt_path.name
            if not sam_path.exists():
                continue 

        # 파일 로드
        try:
            with open(kpt_path, 'r') as f: s_data = json.load(f)
            with open(sam_path, 'r') as f: m_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        # SAM 객체 파싱
        sam_objs = []
        for obj in m_data.get('objects', []):
            try:
                bbox = obj['bbox']
                ctr, area = get_box_details(bbox)
                sam_objs.append({'id': obj['id'], 'bbox': bbox, 'ctr': ctr, 'area': area})
            except: continue

        # ID 매칭
        is_modified = False
        for inst in s_data.get('instance_info', []):
            if 'bbox' not in inst or not inst['bbox']: continue
            try:
                s_box = inst['bbox']
                s_box_coord = s_box[0][:4] if isinstance(s_box[0], (list, tuple)) else s_box[:4]
                s_ctr, s_area = get_box_details(s_box_coord)
            except: continue

            best_id, max_iou, min_dist = None, 0.3, 200
            for obj in sam_objs:
                dist = math.sqrt((s_ctr[0] - obj['ctr'][0])**2 + (s_ctr[1] - obj['ctr'][1])**2)
                if dist > min_dist: continue
                iou = calculate_iou(s_box_coord, obj['bbox'])
                if iou > max_iou:
                    max_iou, best_id = iou, obj['id']
            
            if best_id is not None:
                inst['instance_id'] = best_id
                is_modified = True
        
        # 저장
        if is_modified or (save_dir != kpt_dir):
            output_file = save_dir / kpt_path.name
            with open(output_file, 'w') as f:
                json.dump(s_data, f, indent=4)
            processed_count += 1

    return processed_count