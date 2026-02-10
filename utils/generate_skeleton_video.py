import json
import cv2
import random
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Union

def generate_17kpt_skeleton_video(
    frame_dir: str, 
    kpt_dir: str, 
    output_path: str, 
    conf_threshold: float = 0.0
):
    '''
    frame_dir : frame 위치
    kpt_dir : kpt 위치
    output_path : output 위치
    conf_threshold : conf socore threshold. 해당 숫자 이상의 kpt만 표시.

    프레임 위에 json 내부 skeleton을 덧씌워서 mp4 생성
    '''
    frame_path = Path(frame_dir)
    json_path = Path(kpt_dir)
    save_path = Path(output_path)

    # 1. 경로 및 파일 확인
    if not json_path.exists():
        print(f"❌ JSON 경로 오류: {json_path}")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png")))
    json_files = sorted(list(json_path.glob("*.json")))
    
    if not frame_files:
        print("❌ 프레임 이미지가 없습니다.")
        return

    # 2. 색상 설정 (BGR)
    COLOR_SKELETON = (100, 100, 100) # 뼈대: 회색
    COLOR_RIGHT = (0, 0, 255)        # 오른쪽: Red
    COLOR_LEFT = (255, 0, 0)         # 왼쪽: Blue
    COLOR_ID = (0, 255, 0)           # ID: Green
    COLOR_TEXT = (255, 255, 255)     # 텍스트: White (UI용)
    
    # 3. Skeleton 연결 정보 (COCO 17 Keypoints 표준)
    SKELETON_LINKS = [
        (5, 7), (7, 9),     # 왼팔
        (6, 8), (8, 10),    # 오른팔
        (11, 13), (13, 15), # 왼다리
        (12, 14), (14, 16), # 오른다리
        (5, 6),             # 어깨 사이
        (11, 12),           # 골반 사이
        (5, 11), (6, 12)    # 몸통
    ]

    LEFT_INDICES = {5, 7, 9, 11, 13, 15}
    RIGHT_INDICES = {6, 8, 10, 12, 14, 16}
    
    # 4. 비디오 설정
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    print(f"\n🎬 비디오 생성 시작: {save_path.name}")

    loop_len = len(frame_files)
    
    for i in tqdm(range(loop_len), desc="Rendering Video"):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None: continue

        # =========================================================
        # 🟢 [UI 수정] Frame 번호 (검은 박스 + 흰 글씨 + 현재/전체)
        # =========================================================
        frame_text = f"Frame: {i}/{loop_len}"
        
        # 폰트 설정 (작게)
        font_scale = 0.6
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 텍스트 크기 계산 (배경 박스 사이즈를 위해)
        (text_w, text_h), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
        
        # 박스 좌표 계산 (좌측 하단 여백 20px 기준)
        margin = 20
        box_x1 = margin
        box_y1 = h - margin - text_h - 10  # 글자 위 여백 10
        box_x2 = margin + text_w + 20      # 글자 좌우 여백 합쳐서 20
        box_y2 = h - margin + 5            # 글자 아래 여백 5
        
        # 1. 검은색 배경 박스 그리기 (채우기)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        
        # 2. 하얀색 텍스트 그리기
        cv2.putText(frame, frame_text, (margin + 10, h - margin), font, 
                    font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)
        # =========================================================

        # JSON 로드
        data = {}
        if i < len(json_files):
            try:
                with open(json_files[i], 'r') as f:
                    data = json.load(f)
            except:
                pass
        
        # 인스턴스 그리기
        for inst in data.get('instance_info', []):
            if inst.get('score', 1.0) <= conf_threshold: continue
            
            if 'keypoints' not in inst: continue
            
            # numpy 배열로 변환
            kpts_data = inst['keypoints']
            coords = np.array(kpts_data)
            
            # 점수 가져오기
            if 'keypoint_scores' in inst:
                scores = np.array(inst['keypoint_scores'])
            else:
                scores = np.ones(len(coords))
            
            num_kpts = len(coords)
            obj_id = inst.get('instance_id', inst.get('id', '?'))

            # --- [Step 1] Lines (5~16번만) ---
            for u, v in SKELETON_LINKS:
                if u >= num_kpts or v >= num_kpts: continue
                
                if scores[u] > conf_threshold and scores[v] > conf_threshold:
                    pt1 = (int(coords[u][0]), int(coords[u][1]))
                    pt2 = (int(coords[v][0]), int(coords[v][1]))
                    
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2, cv2.LINE_AA)

            # --- [Step 2] Dots (5~16번만) ---
            for idx in range(num_kpts):
                if not (5 <= idx <= 16): continue
                
                score = scores[idx]
                x, y = int(coords[idx][0]), int(coords[idx][1])

                if score > conf_threshold and x > 0 and y > 0:
                    if idx in RIGHT_INDICES:
                        color = COLOR_RIGHT
                    elif idx in LEFT_INDICES:
                        color = COLOR_LEFT
                    else:
                        color = (0, 255, 0)
                        
                    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)

            # --- [Step 3] BBox & ID ---
            bbox = inst.get('bbox')
            if bbox:
                b = np.array(bbox).flatten()
                if len(b) >= 4:
                    x1, y1, x2, y2 = map(int, b[:4])
                    
                    # ID 박스 (초록색)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ID, 2)
                    
                    # ID 텍스트 라벨 (검은 배경 + 흰 글씨로 변경하여 가독성 확보)
                    label = f"ID: {obj_id}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # 라벨 배경 (초록색)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), COLOR_ID, -1)
                    # 라벨 텍스트 (흰색 - UI와 통일감 또는 검은색 선택 가능)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 0), 1, cv2.LINE_AA) # 글씨는 검은색으로 해서 대비 효과

        out.write(frame)

    out.release()
    print(f"\n✅ 비디오 생성 완료: {save_path}")



# ==============================================================================
# 🛠️ 1. 내부 유틸리티 함수
# ==============================================================================
def rle_to_mask(rle: List[int], height: int, width: int) -> np.ndarray:
    """RLE 리스트를 마스크로 변환"""
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle: 
        return mask.reshape((height, width))
    
    rle = np.array(rle)
    starts = rle[0::2] - 1
    lengths = rle[1::2]
    ends = starts + lengths
    
    for lo, hi in zip(starts, ends):
        lo, hi = max(lo, 0), min(hi, len(mask))
        mask[lo:hi] = 1
        
    return mask.reshape((height, width))

_color_map: Dict[int, List[int]] = {}

def get_color(obj_id: int) -> List[int]:
    """객체 ID별 고유 랜덤 색상 반환"""
    if obj_id not in _color_map:
        # 가독성을 위해 너무 어둡지 않은 색상 생성
        _color_map[obj_id] = [
            random.randint(50, 255), 
            random.randint(50, 255), 
            random.randint(50, 255)
        ]
    return _color_map[obj_id]

# ==============================================================================
# 🎨 2. 시각화 헬퍼 함수 (수정됨)
# ==============================================================================
def draw_mask_on_overlay(overlay: np.ndarray, mask: np.ndarray, obj_id: int):
    """오버레이 레이어에 마스크(색칠)만 그리기"""
    color = get_color(obj_id)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 외곽선과 내부 채우기
    cv2.drawContours(overlay, contours, -1, color, 2) 
    cv2.fillPoly(overlay, contours, color) 

def draw_bbox_and_id(frame: np.ndarray, mask: np.ndarray, obj_id: int):
    """
    최종 프레임에 BBox와 ID 라벨을 선명하게 그리기
    - BBox: 객체 외곽 사각형
    - Label: BBox 좌측 상단에 배경색과 함께 표시
    """
    color = get_color(obj_id)
    y, x = np.where(mask)
    
    if len(y) > 0:
        # BBox 좌표 계산
        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)
        
        # 1. BBox 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 2. ID 라벨 그리기 (BBox 위에)
        label = f"ID: {obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (w_text, h_text), _ = cv2.getTextSize(label, font, scale, thickness)
        
        # 라벨 배경 박스 (BBox 색상과 동일)
        # BBox 위 공간이 부족하면 안쪽에 그림
        if y1 - h_text - 10 > 0:
            box_coords = ((x1, y1 - h_text - 10), (x1 + w_text + 10, y1))
            text_pos = (x1 + 5, y1 - 5)
        else:
            box_coords = ((x1, y1), (x1 + w_text + 10, y1 + h_text + 10))
            text_pos = (x1 + 5, y1 + h_text + 5)

        cv2.rectangle(frame, box_coords[0], box_coords[1], color, -1)
        
        # 텍스트 (흰색으로 고정하여 가독성 확보)
        cv2.putText(frame, label, text_pos, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_frame_counter(frame: np.ndarray, current_idx: int, total_frames: int):
    """좌측 하단에 검은 박스 + 흰 글씨로 프레임 표시"""
    text = f"Frame: {current_idx}/{total_frames}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    img_h, img_w = frame.shape[:2]
    margin = 30
    
    # 박스 좌표 (좌측 하단)
    x1 = margin
    y1 = img_h - margin - h - 10
    x2 = margin + w + 20
    y2 = img_h - margin + 10
    
    # 검은색 배경 박스
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    # 흰색 텍스트
    cv2.putText(frame, text, (margin + 10, img_h - margin), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ==============================================================================
# 🚀 3. 메인 함수
# ==============================================================================
def generate_sam_video(
    frame_dir: Union[str, Path],
    json_dir: Union[str, Path],
    output_path: Union[str, Path],
    fps: float = 30.0,
    alpha: float = 0.5
):
    """
    SAM 결과 JSON을 프레임에 오버레이하여 비디오 생성
    - Mask: 반투명 (Alpha Blending)
    - BBox & ID: 불투명 (Sharp)
    - UI: 좌측 하단 검은 박스
    """
    frame_dir, json_dir, output_path = Path(frame_dir), Path(json_dir), Path(output_path)

    # 1. 파일 확인
    frame_files = sorted(glob.glob(str(frame_dir / "*.jpg")))
    if not frame_files:
        print(f"❌ [Error] 이미지 없음: {frame_dir}")
        return

    # 2. 첫 프레임 정보
    first_img = cv2.imread(frame_files[0])
    if first_img is None:
        print(f"❌ [Error] 첫 프레임 로드 실패")
        return
    h, w = first_img.shape[:2]

    print(f"🎬 [SAM Video] 시작: {w}x{h} | {len(frame_files)} frames | FPS {fps}")
    
    # 3. Writer 설정
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 4. 루프 실행
    for i, img_path in enumerate(tqdm(frame_files, desc="🚀 Rendering", unit="frame")):
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        overlay = frame.copy() # 마스크용 레이어
        detected_objects = []  # (mask, id) 저장용 리스트
        
        # JSON 로드
        json_path = json_dir / f"{i:06d}.json"
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if "objects" in data:
                    for obj in data["objects"]:
                        rle = obj.get("segmentation", {}).get("counts")
                        if rle:
                            mask = rle_to_mask(rle, h, w)
                            if mask.sum() > 0:
                                obj_id = obj.get("id")
                                # 1. 마스크는 overlay 레이어에 그리기 (투명도 적용 예정)
                                draw_mask_on_overlay(overlay, mask, obj_id)
                                # 2. BBox를 위해 정보 저장
                                detected_objects.append((mask, obj_id))
            except:
                pass

        # 5. 마스크 합성 (Alpha Blending)
        if detected_objects:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # 6. BBox 및 ID 그리기 (합성된 프레임 위에 선명하게 덧그리기)
            for mask, obj_id in detected_objects:
                draw_bbox_and_id(frame, mask, obj_id)
        
        # 7. 프레임 카운터 (좌측 하단 검은 박스)
        draw_frame_counter(frame, i, len(frame_files))

        out.write(frame)

    out.release()
    print(f"🎉 [완료] 저장됨: {output_path}\n")


# =================================================================
# 🎨 색상 및 연결 정보 설정 (133 Keypoints)
# =================================================================
# BGR Colors
COLOR_BODY = (255, 100, 0)   # 몸통 (Blue-ish)
COLOR_FACE = (0, 255, 255)   # 얼굴 (Yellow)
COLOR_L_HAND = (0, 0, 255)   # 왼손 (Red)
COLOR_R_HAND = (255, 0, 0)   # 오른손 (Blue)
COLOR_L_FOOT = (0, 128, 255) # 왼발 (Orange)
COLOR_R_FOOT = (255, 128, 0) # 오른발 (Cyan)
COLOR_ID = (0, 255, 0)       # ID 박스 (Green)
COLOR_TEXT = (255, 255, 255)

# COCO-WholeBody 133 Keypoints Links
SKELETON_LINKS_133 = {
    # Body (0~16)
    'body': [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), 
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    ],
    # Feet (17~22)
    'feet': [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)],
    # Face (23~90)
    'face': [
        (23, 24), (24, 25), (26, 27), (27, 28), 
        (62, 63), (63, 64), (64, 65), (65, 66)
    ],
    # Left Hand (91~111)
    'l_hand': [
        (9, 91), (91, 92), (92, 93), (93, 94), (94, 95),
        (91, 96), (96, 97), (97, 98), (98, 99),
        (91, 100), (100, 101), (101, 102), (102, 103),
        (91, 104), (104, 105), (105, 106), (106, 107),
        (91, 108), (108, 109), (109, 110), (110, 111)
    ],
    # Right Hand (112~132)
    'r_hand': [
        (10, 112), (112, 113), (113, 114), (114, 115), (115, 116),
        (112, 117), (117, 118), (118, 119), (119, 120),
        (112, 121), (121, 122), (122, 123), (123, 124),
        (112, 125), (125, 126), (126, 127), (127, 128),
        (112, 129), (129, 130), (130, 131), (131, 132)
    ]
}

def generate_133kpt_skeleton_video(
    frame_dir: str, 
    kpt_dir: str, 
    output_path: str, 
    conf_threshold: float = 0.05 # 점수가 낮으므로 기본값을 낮게 설정
):
    frame_path = Path(frame_dir)
    json_path = Path(kpt_dir)
    save_path = Path(output_path)

    if not json_path.exists():
        print(f"❌ JSON 경로 오류: {json_path}")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png")))
    json_files = sorted(list(json_path.glob("*.json")))
    
    if not frame_files:
        print("❌ 프레임 이미지가 없습니다.")
        return

    # Video Init
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    print(f"\n🎬 133-KPT 비디오 생성 시작: {save_path.name}")
    loop_len = len(frame_files)
    
    for i in tqdm(range(loop_len), desc="Rendering 133-KPT"):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None: continue
        
        # UI: Frame Counter
        draw_frame_counter(frame, i, loop_len)

        # JSON Load
        data = {}
        if i < len(json_files):
            try:
                with open(json_files[i], 'r') as f: data = json.load(f)
            except: pass
        
        # Instance Loop
        for inst in data.get('instance_info', []):
            # score 필드가 있으면 체크, 없으면 pass
            if inst.get('score', 1.0) <= 0.0: continue 
            if 'keypoints' not in inst: continue
            
            # --- [핵심 변경] 데이터 파싱 ---
            # keypoints: [[x, y, score], [x, y, score], ...] (133 x 3)
            raw_kpts = np.array(inst['keypoints'])
            
            if raw_kpts.shape[1] >= 2:
                coords = raw_kpts[:, :2] # (133, 2)
                
                # 점수 정보 가져오기
                if raw_kpts.shape[1] >= 3:
                    scores = raw_kpts[:, 2]
                elif 'keypoint_scores' in inst:
                    scores = np.array(inst['keypoint_scores'])
                else:
                    scores = np.ones(len(coords))
            else:
                continue # 형식이 맞지 않으면 건너뜀

            obj_id = inst.get('instance_id', inst.get('id', '?'))
            
            # --- Draw Links ---
            for part, links in SKELETON_LINKS_133.items():
                if part == 'body': color = COLOR_BODY
                elif part == 'face': color = COLOR_FACE
                elif part == 'l_hand': color = COLOR_L_HAND
                elif part == 'r_hand': color = COLOR_R_HAND
                else: color = COLOR_BODY
                
                for u, v in links:
                    if u >= len(coords) or v >= len(coords): continue
                    
                    # 두 점 모두 임계값 이상이고 좌표가 유효할 때만 그리기
                    if scores[u] > conf_threshold and scores[v] > conf_threshold:
                        pt1 = (int(coords[u][0]), int(coords[u][1]))
                        pt2 = (int(coords[v][0]), int(coords[v][1]))
                        
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

            # --- Draw Dots ---
            for idx, (x, y) in enumerate(coords):
                if scores[idx] > conf_threshold and x > 0 and y > 0:
                    if 0 <= idx < 17: color = COLOR_BODY
                    elif 17 <= idx < 23: color = COLOR_L_FOOT
                    elif 23 <= idx < 91: color = COLOR_FACE
                    elif 91 <= idx < 112: color = COLOR_L_HAND
                    elif 112 <= idx < 133: color = COLOR_R_HAND
                    else: color = (200, 200, 200)
                    
                    cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)

            # --- Draw ID & BBox ---
            bbox = inst.get('bbox')
            if bbox:
                b = np.array(bbox).flatten()
                if len(b) >= 4:
                    x1, y1, x2, y2 = map(int, b[:4])
                    
                    # ID Box (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ID, 2)
                    
                    label = f"ID: {obj_id}"
                    (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # Label Background
                    cv2.rectangle(frame, (x1, y1 - h_txt - 10), (x1 + w_txt + 10, y1), COLOR_ID, -1)
                    
                    # Label Text
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0,0,0), 1, cv2.LINE_AA) # 검은 글씨

        out.write(frame)

    out.release()
    print(f"✅ 133-KPT 비디오 완료: {save_path}")

def draw_frame_counter_seg(frame, real_idx, seg_idx, total_seg_frames):
    """구간 정보가 포함된 프레임 카운터"""
    text = f"Frame: {real_idx} (Seg: {seg_idx}/{total_seg_frames})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    img_h, img_w = frame.shape[:2]
    margin = 20
    
    x1, y1 = margin, img_h - margin - h - 10
    x2, y2 = margin + w + 20, img_h - margin + 5
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, text, (margin + 10, img_h - margin), font, scale, COLOR_TEXT, thickness, cv2.LINE_AA)

# =================================================================
# 🚀 3. 구간 비디오 생성 함수 (17-KPT)
# =================================================================
def generate_segment_video_17kpt(
    frame_dir, 
    kpt_dir, 
    output_path, 
    start_idx, 
    end_idx, 
    conf_threshold=0.0
):
    """
    17 Keypoints 스타일로 특정 구간의 비디오를 생성합니다.
    """
    frame_path = Path(frame_dir)
    json_path = Path(kpt_dir)
    save_path = Path(output_path)

    # 1. 파일 리스트 로드
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png")))
    
    if not frame_files:
        print("❌ 프레임 이미지가 없습니다.")
        return

    # 2. 구간 클리핑
    start_idx = max(0, start_idx)
    end_idx = min(len(frame_files), end_idx)
    if start_idx >= end_idx:
        print("⚠️ 유효하지 않은 구간입니다.")
        return

    target_frames = frame_files[start_idx:end_idx]
    
    # 3. 비디오 설정
    save_path.parent.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(target_frames[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    print(f"🎬 17-KPT Segment 생성: {save_path.name}")
    print(f"   Range: {start_idx} ~ {end_idx} ({len(target_frames)} frames)")

    # 4. 렌더링 루프
    for i, frame_file in enumerate(tqdm(target_frames, desc="Rendering Segment")):
        # 프레임 로드
        frame = cv2.imread(str(frame_file))
        if frame is None: continue

        # 실제 프레임 인덱스 (파일명 기준 추정)
        try:
            real_idx = int(frame_file.stem)
        except:
            real_idx = start_idx + i

        # UI 표시
        draw_frame_counter_seg(frame, real_idx, i, len(target_frames))

        # JSON 매칭
        current_json = json_path / f"{frame_file.stem}.json"
        
        if current_json.exists():
            try:
                with open(current_json, 'r') as f:
                    data = json.load(f)
                
                # 인스턴스 그리기
                for inst in data.get('instance_info', []):
                    if inst.get('score', 1.0) <= conf_threshold: continue
                    if 'keypoints' not in inst: continue

                    # 데이터 파싱
                    raw_kpts = np.array(inst['keypoints'])
                    if raw_kpts.shape[1] < 2: continue

                    coords = raw_kpts[:, :2]
                    # 점수 파싱 (3번째 값이 있거나 별도 리스트)
                    if raw_kpts.shape[1] >= 3:
                        scores = raw_kpts[:, 2]
                    elif 'keypoint_scores' in inst:
                        scores = np.array(inst['keypoint_scores'])
                    else:
                        scores = np.ones(len(coords))

                    obj_id = inst.get('instance_id', inst.get('id', '?'))
                    num_kpts = len(coords)

                    # --- Draw Lines (Skeleton) ---
                    for u, v in SKELETON_LINKS_17:
                        if u >= num_kpts or v >= num_kpts: continue
                        
                        if scores[u] > conf_threshold and scores[v] > conf_threshold:
                            pt1 = (int(coords[u][0]), int(coords[u][1]))
                            pt2 = (int(coords[v][0]), int(coords[v][1]))
                            
                            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2, cv2.LINE_AA)

                    # --- Draw Dots (Keypoints) ---
                    for idx, (x, y) in enumerate(coords):
                        # 5번~16번 (몸통) 위주로 시각화 (기존 로직 유지)
                        if not (5 <= idx <= 16): continue
                        
                        if scores[idx] > conf_threshold and x > 0 and y > 0:
                            if idx in RIGHT_INDICES:
                                color = COLOR_RIGHT
                            elif idx in LEFT_INDICES:
                                color = COLOR_LEFT
                            else:
                                color = (0, 255, 0)
                            
                            cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)

                    # --- Draw ID & BBox ---
                    bbox = inst.get('bbox')
                    if bbox:
                        b = np.array(bbox).flatten()
                        if len(b) >= 4:
                            x1, y1, x2, y2 = map(int, b[:4])
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ID, 2)
                            
                            label = f"ID: {obj_id}"
                            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            
                            cv2.rectangle(frame, (x1, y1 - h_txt - 10), (x1 + w_txt + 10, y1), COLOR_ID, -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, (0,0,0), 1, cv2.LINE_AA)

            except Exception as e:
                # print(f"JSON Error: {e}")
                pass

        out.write(frame)

    out.release()
    print(f"✅ 완료: {save_path}")