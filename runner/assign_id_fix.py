import cv2 # OpenCV 라이브러리 (시각화용)
import json # JSON 입출력
import random # 마스크 랜덤 색상용
import numpy as np # 수치 및 배열 연산
from pathlib import Path # 안전한 경로 관리
from tqdm import tqdm # 진행 상태바 출력

KPT_WEIGHTS = { # 관절별 중요도 가중치
    0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, # 얼굴 부위
    5: 2.0, 6: 2.0,                         # 양 어깨
    7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0,        # 양 팔
    11: 2.0, 12: 2.0,                       # 양 골반
    13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0      # 양 다리
}

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

def get_bbox_from_mask(mask): # 마스크에서 Min-Max Bbox 추출
    coords = np.argwhere(mask > 0) # 1인 픽셀 좌표 모두 찾기
    if coords.size == 0: return None # 비어있으면 종료
    
    y_min, x_min = coords.min(axis=0) # 좌측 상단 최소 좌표
    y_max, x_max = coords.max(axis=0) # 우측 하단 최대 좌표
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)] # [x1, y1, x2, y2] 반환

def calculate_iou(box1, box2): # IoU 계산 함수
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1]) # 겹친 왼쪽 위
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3]) # 겹친 오른쪽 아래
    if x_right < x_left or y_bottom < y_top: return 0.0 # 안 겹치면 0
    inter_area = (x_right - x_left) * (y_bottom - y_top) # 교집합 면적
    area1, area2 = (box1[2]-box1[0])*(box1[3]-box1[1]), (box2[2]-box2[0])*(box2[3]-box2[1]) # 각 면적
    return inter_area / float(area1 + area2 - inter_area) # IoU 결과값

def filter_duplicate_skeletons(instance_info, iou_threshold=0.85):
    """
    IoU가 threshold(예: 85%) 이상 겹치는 스켈레톤 중 
    키포인트 신뢰도 합(conf_score)이 더 높은 하나만 남깁니다.
    """
    if not instance_info: return []

    # 1. 각 인스턴스에 총 신뢰도 점수(total_score) 계산해서 임시 저장
    for inst in instance_info:
        inst['total_score'] = sum(inst.get('keypoint_scores', []))
    
    # 2. 점수가 높은 순으로 내림차순 정렬 (높은 걸 먼저 살리기 위함)
    instance_info.sort(key=lambda x: x['total_score'], reverse=True)
    
    filtered_info = []
    
    # 3. NMS 수행
    for inst in instance_info:
        # Bbox 평탄화 (2차원일 경우 대비)
        raw_box = np.array(inst.get('bbox', [])).flatten().tolist()
        box_a = raw_box[:4] if len(raw_box) >= 4 else [0,0,0,0]
        
        keep = True
        for keep_inst in filtered_info:
            raw_box_k = np.array(keep_inst.get('bbox', [])).flatten().tolist()
            box_b = raw_box_k[:4] if len(raw_box_k) >= 4 else [0,0,0,0]
            
            # IoU가 기준치 이상으로 겹치면 (같은 사람이라고 판단되면) 버림
            if calculate_iou(box_a, box_b) > iou_threshold:
                keep = False 
                break
                
        if keep:
            # 보관하기로 결정된 스켈레톤은 불필요한 total_score 키 제거 후 리스트에 추가
            inst.pop('total_score', None)
            filtered_info.append(inst)
            
    return filtered_info

def assign_sam_ids_to_keypoints(sam_dir, kpt_dir, output_dir=None): # 매칭 메인 함수
    save_dir = Path(output_dir) if output_dir else Path(kpt_dir) # 저장 폴더 설정 
    save_dir.mkdir(parents=True, exist_ok=True) # 폴더 생성 
    processed_count = 0 # 완료 횟수 카운터 초기화 
    
    prev_s_data = None # [추가] 이전 프레임 데이터를 저장할 변수 초기화

    for kpt_path in tqdm(sorted(Path(kpt_dir).glob("*.json")), desc="Aligning"): # 파일 순회 (이름순 정렬 필수!)
        sam_path = Path(sam_dir) / f"sam_{kpt_path.name}" # SAM 파일 경로 
        if not sam_path.exists(): sam_path = Path(sam_dir) / kpt_path.name # 예외 이름 체크 
        if not sam_path.exists(): continue # 파일 없으면 스킵 

        with open(kpt_path, 'r') as f: s_data = json.load(f) # 스켈레톤 로드 
        with open(sam_path, 'r') as f: m_data = json.load(f) # SAM 로드 
        
        s_data['instance_info'] = filter_duplicate_skeletons(s_data.get('instance_info', []))

        sam_info_list = [] # SAM 정보 리스트 
        for obj in m_data.get('objects', []): # SAM 객체 반복 
            mask = rle_to_mask(obj.get('segmentation', {})) # 마스크 복원 
            if mask is not None: # 복원 성공 시 
                x1, y1, x2, y2 = get_bbox_from_mask(mask) # Min-Max 추출
                sam_info_list.append({'id': obj['id'], 'mask': mask, 'bbox': [x1, y1, x2, y2], 'area': (x2-x1)*(y2-y1)}) # 정보 저장

        is_modified = False # 수정 여부 플래그 
        for inst_idx, inst in enumerate(s_data.get('instance_info', [])): # 사람별 반복 (인덱스 활용 위해 enumerate로 변경)
            kpts = inst.get('keypoints', []) # 좌표 리스트
            kpt_scores = inst.get('keypoint_scores', []) # 점수 리스트
            
            raw_box = np.array(inst.get('bbox', [])).flatten().tolist() # Bbox 평탄화
            sx1, sy1, sx2, sy2 = map(int, raw_box[:4]) # 정수 변환
            inst['bbox'] = [sx1, sy1, sx2, sy2] # 선제적 포맷 변경
            skel_area = (sx2 - sx1) * (sy2 - sy1) # 면적 계산

            # [추가] 이전 프레임의 동일 인덱스 인스턴스 정보 가져오기
            prev_inst = None
            if prev_s_data and inst_idx < len(prev_s_data.get('instance_info', [])):
                prev_inst = prev_s_data['instance_info'][inst_idx]

            votes = {} # 투표함 초기화 
            for k_idx, (x_val, y_val) in enumerate(kpts): # 좌표 반복
                conf = kpt_scores[k_idx] if k_idx < len(kpt_scores) else 0.0 # 신뢰도 가져오기
                if conf < 0.01: continue # 신뢰도 낮으면 제외
                
                x, y = int(x_val), int(y_val) # 정수 좌표 
                for m_idx, m_info in enumerate(sam_info_list): # 마스크 대조 
                    mh, mw = m_info['mask'].shape # 마스크 크기 
                    if 0 <= x < mw and 0 <= y < mh and m_info['mask'][y, x] == 1: # 내부에 있다면 
                        score = KPT_WEIGHTS.get(k_idx, 1.0) * conf # 점수 계산 
                        votes[m_idx] = votes.get(m_idx, 0) + score # 점수 누적 

            if votes: # 투표 결과 있는 경우 (Plan A) 
                # [수정] 점수 계산 로직 분리 및 시계열 가중치 추가
                def scoring_function(m_idx):
                    m_info = sam_info_list[m_idx]
                    
                    # 1. 기본 점수 (투표수 * 면적 유사도)
                    area_ratio = min(skel_area, m_info['area']) / max(skel_area, m_info['area'])
                    base_score = votes[m_idx] * area_ratio
                    
                    # 2. 시계열 가중치 (이전 프레임과의 IoU 기반)
                    temporal_weight = 1.0
                    if prev_inst and 'bbox' in prev_inst:
                        iou_with_prev = calculate_iou(prev_inst['bbox'], m_info['bbox'])
                        temporal_weight += iou_with_prev * 0.2  # IoU가 1이면 최대 20% 보너스 점수
                    
                    return base_score * temporal_weight

                best_m_idx = max(votes, key=scoring_function) # 새로운 점수 함수로 최고점 산출
                chosen = sam_info_list[best_m_idx] # 마스크 선정 
                inst['instance_id'], inst['bbox'] = int(chosen['id']), chosen['bbox'] # 정보 갱신 
                is_modified = True # 수정 기록 
            elif sam_info_list: # 투표 실패 시 IoU 매칭 (Plan B)
                best_iou, best_m_idx = 0.1, -1 # IoU 기준치
                for m_idx, m_info in enumerate(sam_info_list): # 마스크 순회
                    iou = calculate_iou(inst['bbox'], m_info['bbox']) # IoU 계산
                    if iou > best_iou: best_iou, best_m_idx = iou, m_idx # 최고치 갱신
                if best_m_idx != -1: # 매칭 성공 시
                    inst['instance_id'], inst['bbox'] = int(sam_info_list[best_m_idx]['id']), sam_info_list[best_m_idx]['bbox'] # 갱신
                    is_modified = True # 수정 기록

        if is_modified or (save_dir != Path(kpt_dir)): # 저장 조건 충족 시 
            with open(save_dir / kpt_path.name, 'w') as f: json.dump(s_data, f, indent=4) # 파일 저장 
            processed_count += 1 # 완료 횟수 증가 
        
        # [추가] 다음 루프를 위해 현재 프레임 데이터를 저장
        prev_s_data = s_data
            
    return processed_count # 최종 완료 횟수 반환

def debug_frame_to_txt(sam_dir, kpt_dir, target_frame="000160", txt_path="./a.txt"): # 디버그 텍스트 출력
    kpt_path = Path(kpt_dir) / f"{target_frame}.json" # KPT 경로
    sam_path = Path(sam_dir) / f"sam_{target_frame}.json" # SAM 경로
    if not sam_path.exists(): sam_path = Path(sam_dir) / f"{target_frame}.json" # 경로 재확인
    if not kpt_path.exists() or not sam_path.exists(): return # 누락 방지
    
    with open(kpt_path, 'r') as f: s_data = json.load(f) # KPT 메모리 로드
    with open(sam_path, 'r') as f: m_data = json.load(f) # SAM 메모리 로드
    
    with open(txt_path, 'w', encoding='utf-8') as out_f: # 텍스트 쓰기 오픈
        out_f.write(f"=== [프레임 {target_frame} 디버그] ===\n\n") # 헤더 작성
        
        out_f.write("[1] SAM 마스크 (Min-Max)\n") # SAM 섹션
        for obj in m_data.get('objects', []): # SAM 반복
            mask = rle_to_mask(obj.get('segmentation', {})) # 복원
            if mask is not None: # 정상 처리
                exact_bbox = get_bbox_from_mask(mask) # 추출
                # Bbox 있으면 면적 계산, 없으면 0
                area = (exact_bbox[2]-exact_bbox[0]) * (exact_bbox[3]-exact_bbox[1]) if exact_bbox else 0
                out_f.write(f" - ID: {obj.get('id')}, Bbox: {exact_bbox}, 면적: {area}\n") # 기록
                
        out_f.write("\n[2] 스켈레톤 (Min-Max)\n") # KPT 섹션
        for idx, inst in enumerate(s_data.get('instance_info', [])): # KPT 반복
            raw_bbox = np.array(inst.get('bbox', [])).flatten().tolist() # 리스트 평탄화
            out_f.write(f" - 사람 #{idx+1} Bbox: {raw_bbox[:4]}\n") # 기록

def debug_frame_to_png(frame_dir, sam_dir, kpt_dir, target_frame="000160", output_png_path="./debug.png"): # 디버그 시각화
    img_path = next(Path(frame_dir).glob(f"{target_frame}.*"), None) # 이미지 동적 찾기
    if not img_path: return # 없으면 종료
    
    kpt_path = Path(kpt_dir) / f"{target_frame}.json" # KPT 경로
    sam_path = Path(sam_dir) / f"sam_{target_frame}.json" # SAM 경로
    if not sam_path.exists(): sam_path = Path(sam_dir) / f"{target_frame}.json" # 이름 다를 시 체크
    
    if not kpt_path.exists() or not sam_path.exists(): return # 누락 방지
    
    img = cv2.imread(str(img_path)) # 이미지 배열화
    if img is None: return # 읽기 에러 방지
    
    with open(kpt_path, 'r') as f: s_data = json.load(f) # KPT 로드
    with open(sam_path, 'r') as f: m_data = json.load(f) # SAM 로드
    
    overlay = img.copy() # 마스크 합성용 투명층
    for obj in m_data.get('objects', []): # 마스크 반복
        mask = rle_to_mask(obj.get('segmentation', {})) # 복원
        if mask is not None: # 성공 시
            color = [random.randint(0, 255) for _ in range(3)] # BGR 랜덤 색상
            overlay[mask == 1] = color # 마스크 칠하기
            exact_bbox = get_bbox_from_mask(mask) # 추출
            if exact_bbox: # 정상 시
                x1, y1, x2, y2 = exact_bbox # 좌표 분류
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # 파란 Bbox
                
    for inst in s_data.get('instance_info', []): # 관절점 반복
        raw_bbox = np.array(inst.get('bbox', [])).flatten().tolist() # 평탄화
        if len(raw_bbox) >= 4: # 스켈레톤 Bbox
            sx1, sy1, sx2, sy2 = map(int, raw_bbox[:4]) # 정수 캐스팅
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 0, 255), 1) # 빨간 얇은 Bbox
            
        for kpt in inst.get('keypoints', []): # 개별 점 반복
            if len(kpt) >= 3 and kpt[2] > 0.1: # 신뢰도 이상만
                cv2.circle(img, (int(kpt[0]), int(kpt[1])), 4, (0, 0, 255), -1) # 빨간 점
                
    final_img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0) # 반투명 알파 블렌딩
    cv2.imwrite(output_png_path, final_img) # 파일 저장

def debug_mask_pixel_check(sam_dir, kpt_dir, target_frame="000400"): # 특정 프레임의 픽셀 매칭을 정밀 분석합니다.
    kpt_path = Path(kpt_dir) / f"{target_frame}.json" # 키포인트 파일 경로 설정
    sam_path = Path(sam_dir) / f"sam_{target_frame}.json" # SAM 파일 경로 설정 (sam_ 접두사 포함)
    if not sam_path.exists(): sam_path = Path(sam_dir) / f"{target_frame}.json" # 접두사 없을 경우 대비

    if not kpt_path.exists() or not sam_path.exists(): # 파일 존재 여부 확인
        print(f"🚨 [파일 없음] {target_frame} 프레임 데이터를 찾을 수 없습니다.") # 에러 메시지
        return

    with open(kpt_path, 'r') as f: s_data = json.load(f) # 스켈레톤 데이터 로드
    with open(sam_path, 'r') as f: m_data = json.load(f) # SAM 데이터 로드

    print(f"\n" + "="*60) # 구분선
    print(f"🔬 [프레임 {target_frame} 픽셀 정밀 분석 보고서]") # 제목
    print("="*60)

    # 1. SAM 마스크 복원 및 정보 요약
    sam_masks = [] # 복원된 마스크 정보를 담을 리스트
    for obj in m_data.get('objects', []): # SAM 객체 순회
        mask = rle_to_mask(obj.get('segmentation', {})) # RLE를 2차원 배열로 복원
        if mask is not None: # 복원 성공 시
            bbox = get_bbox_from_mask(mask) # Min-Max Bbox 계산
            sam_masks.append({'id': obj['id'], 'mask': mask, 'bbox': bbox}) # 리스트 추가
            print(f"✅ SAM 마스크 ID [{obj['id']}] 복원 완료 (Bbox: {bbox})") # 정보 출력

    # 2. 스켈레톤 인스턴스별 픽셀 히트 테스트
    for p_idx, inst in enumerate(s_data.get('instance_info', [])): # 사람별 반복
        print(f"\n👤 사람 인스턴스 #{p_idx + 1} 분석 시작") # 인스턴스 구분
        kpts = np.array(inst.get('keypoints', [])) # 관절점 배열화
        kpt_scores = inst.get('keypoint_scores', []) # 신뢰도 점수 (별도 필드일 경우)

        for k_idx, kpt in enumerate(kpts): # 17개 관절점 순회
            x, y = int(kpt[0]), int(kpt[1]) # 정수 좌표 변환
            # 신뢰도 값 결정 (배열 내에 있거나 별도 필드에 있을 경우 처리)
            conf = kpt[2] if len(kpt) > 2 else (kpt_scores[k_idx] if k_idx < len(kpt_scores) else 0.0)
            
            status_msg = f"  📍 [{k_idx:2d}] 좌표:({x:4d}, {y:4d}) | 신뢰도:{conf:.4f}" # 기본 정보
            
            if conf < 0.01: # 우리가 설정한 최소 신뢰도 미달 시
                print(f"{status_msg} -> ❌ [점수 미달로 제외]") # 제외 메시지
                continue

            # 모든 마스크와 대조
            hit_found = False # 적중 여부 플래그
            for m_info in sam_masks: # 마스크 리스트 순회
                h, w = m_info['mask'].shape # 마스크 크기 확인
                if 0 <= x < w and 0 <= y < h: # 이미지 범위 내에 있는지 확인
                    pixel_val = m_info['mask'][y, x] # 해당 좌표의 픽셀 값 확인 (0 또는 1)
                    if pixel_val == 1: # 마스크 내부라면
                        weight = KPT_WEIGHTS.get(k_idx, 1.0) # 부위별 가중치 가져오기
                        print(f"{status_msg} -> 🎯 [HIT!] 마스크 ID:{m_info['id']} (가중치:{weight})") # 적중 메시지
                        hit_found = True # 플래그 업데이트
                
            if not hit_found: # 어떤 마스크에도 속하지 못한 경우
                print(f"{status_msg} -> 💨 [MISS] 마스크 영역 바깥에 찍힘") # 실패 메시지

    print("\n" + "="*60 + "\n") # 하단 구분선



# 실행 예시:
# assign_sam_ids_to_keypoints("./sam_data", "./kpt_data", "./output")
# debug_frame_to_png("./frames", "./sam_data", "./output", "000160")

# sapiens로 뽑은 kpt를 사용하여 counting

# sapiens로 뽑은 kpt를 사용하여 counting

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =================================================================
# 1. 경로 설정 및 모듈 임포트
# =================================================================
print("📋 [Step 0] 초기 설정 및 모듈 로드")
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data.csv"

sys.path.append(str(BASE_DIR))

try:
    from utils.path_list import path_list
    from utils.parser import parse_common_path
    from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video, generate_filtered_id_skeleton_video
    print("✅ 모듈 로드 완료.")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
bosanjin_df = pd.read_csv(BOSANJIN_PATH)

paths = path_list(df.loc[66,'common_path'])
output_path = paths['test']/"assigned_kpt"

assign_sam_ids_to_keypoints(
    sam_dir=paths['sam'],
    kpt_dir=paths['keypoint'],
    output_dir=output_path
)

generate_filtered_id_skeleton_video(
    frame_dir=paths['frame'],
    kpt_dir=output_path,
    output_path=str(paths['test']/"assigned_kpt_video.mp4"),
    target_ids=[1]
)
