import json
import cv2
import random
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Callable
from utils.extract_kpt import normalize_skeleton_array

# ==============================================================================
# ⚙️ 1. 통합 설정 (Configuration)
# ==============================================================================
class Config:
    # 색상 (BGR)
    COLOR_SKELETON = (100, 100, 100)
    COLOR_ID = (0, 255, 0)       # Green
    COLOR_TEXT = (255, 255, 255) # White
    
    # 17 Keypoints (Body) 색상
    COLOR_RIGHT = (0, 0, 255)    # Red
    COLOR_LEFT = (255, 0, 0)     # Blue

    # 133 Keypoints (WholeBody) 색상
    COLOR_BODY = (255, 100, 0)
    COLOR_FACE = (0, 255, 255)
    COLOR_HAND_L = (0, 0, 255)
    COLOR_HAND_R = (255, 0, 0)
    COLOR_FOOT_L = (0, 128, 255)
    COLOR_FOOT_R = (255, 128, 0)

    # 폰트 설정
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1

    # 17 Keypoints 연결 정보 (COCO)
    LINKS_17 = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
        (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
    ]
    KPT_17_LEFT = {5, 7, 9, 11, 13, 15}
    KPT_17_RIGHT = {6, 8, 10, 12, 14, 16}

    # 133 Keypoints 연결 정보 (WholeBody)
    LINKS_133 = {
        'body': [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        'feet': [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)],
        'face': [(23, 24), (24, 25), (26, 27), (27, 28), (62, 63), (63, 64), (64, 65), (65, 66)],
        'l_hand': [(9, 91), (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98), (98, 99), (91, 100), (100, 101), (101, 102), (102, 103), (91, 104), (104, 105), (105, 106), (106, 107), (91, 108), (108, 109), (109, 110), (110, 111)],
        'r_hand': [(10, 112), (112, 113), (113, 114), (114, 115), (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123), (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131), (131, 132)]  
    }

    LINKS_12 = [
        (0, 2), (2, 4), # 왼쪽 팔 (Shoulder-Elbow-Wrist)
        (1, 3), (3, 5), # 오른쪽 팔
        (6, 8), (8, 10),# 왼쪽 다리 (Hip-Knee-Ankle)
        (7, 9), (9, 11),# 오른쪽 다리
        (0, 1),         # 어깨 가로선
        (6, 7),         # 골반 가로선
        (0, 6), (1, 7)  # 몸통 세로선 (어깨-골반)
    ]

    # 왼쪽/오른쪽 구분을 위한 인덱스 집합 (12포인트 기준)
    KPT_12_LEFT = {0, 2, 4, 6, 8, 10}  # 짝수 (기존 5, 7, 9, 11, 13, 15)
    KPT_12_RIGHT = {1, 3, 5, 7, 9, 11} # 홀수 (기존 6, 8, 10, 12, 14, 16)

# ==============================================================================
# 🎨 2. 그리기 유틸리티 (Visualization Utils)
# ==============================================================================
class Visualizer:
    @staticmethod
    def draw_text_with_bg(img: np.ndarray, text: str, pos: tuple, 
                          bg_color=(0, 0, 0), txt_color=(255, 255, 255), align_bottom=False):
        """텍스트 뒤에 배경 박스를 그려 가독성을 높입니다."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        
        if align_bottom:
            box_coords = ((x, y - h - 10), (x + w + 20, y + 5))
            text_pos = (x + 10, y)
        else:
            box_coords = ((x, y - h - 5), (x + w + 10, y + 5))
            text_pos = (x + 5, y)

        cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, -1)
        cv2.putText(img, text, text_pos, font, scale, txt_color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_bbox_and_id(img: np.ndarray, bbox: list, obj_id: Union[int, str], color: tuple):
        """BBox와 ID를 그립니다. (이중 리스트 대응 수정 완료)"""
        if bbox is None: return

        # ⭐️ [핵심 수정] 어떤 형태의 bbox가 들어오든 1차원 배열로 평탄화 (Flatten)
        # 예: [[100, 100, 200, 200]] -> [100, 100, 200, 200]
        bbox_flat = np.array(bbox).flatten()
        
        if len(bbox_flat) < 4: return
        
        # 좌표 변환
        x1, y1, x2, y2 = map(int, bbox_flat[:4])
        
        # 좌표 유효성 확보 (음수 방지 및 순서 정렬)
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        # 박스 그리기
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        
        # ID 텍스트 그리기
        Visualizer.draw_text_with_bg(img, f"ID: {obj_id}", (xmin, ymin), bg_color=color)

    @staticmethod
    def rle_to_mask(rle: List[int], height: int, width: int) -> np.ndarray:
        # ... (기존과 동일) ...
        mask = np.zeros(height * width, dtype=np.uint8)
        if not rle: return mask.reshape((height, width))
        rle = np.array(rle)
        starts = rle[0::2] - 1
        lengths = rle[1::2]
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[max(lo, 0):min(hi, len(mask))] = 1
        return mask.reshape((height, width))
    
# ==============================================================================
# 🚀 3. 핵심 엔진 (Video Engine) - 중복 제거의 핵심!
# ==============================================================================
def create_video_engine(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    json_dir: Union[str, Path],
    draw_callback: Callable[[np.ndarray, dict], np.ndarray],
    fps: int = 30,
    start_idx: int = 0,
    end_idx: Optional[int] = None
):
    """
    모든 비디오 생성 함수의 공통 로직을 처리하는 엔진입니다.
    - draw_callback: 실제 그리기 로직이 담긴 함수를 전달받아 실행합니다.
    """
    frame_path = Path(frame_dir)
    save_path = Path(output_path)
    json_path = Path(json_dir)

    # 1. 파일 리스트 확보
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png")))
    if not frame_files:
        print(f"❌ [Error] 프레임 이미지가 없습니다: {frame_dir}")
        return

    # 2. 구간 설정 (Segment 기능 통합)
    end_idx = min(len(frame_files), end_idx) if end_idx is not None else len(frame_files)
    start_idx = max(0, start_idx)
    target_frames = frame_files[start_idx:end_idx]
    
    if not target_frames:
        print("❌ [Error] 처리할 프레임 구간이 유효하지 않습니다.")
        return

    # 3. 비디오 Writer 초기화
    save_path.parent.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(target_frames[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"🎬 비디오 생성: {save_path.name} (Range: {start_idx}~{end_idx}, Frames: {len(target_frames)})")

    # 4. 렌더링 루프
    for i, frame_file in enumerate(tqdm(target_frames, desc="Processing")):
        frame = cv2.imread(str(frame_file))
        if frame is None: continue

        # JSON 로드
        json_file = json_path / f"{frame_file.stem}.json"
        data = {}
        if json_file.exists():
            try:
                with open(json_file, 'r') as f: data = json.load(f)
            except: pass

        # ⭐️ 콜백 실행 (실제 그리기)
        frame = draw_callback(frame, data)

        # 공통 UI: 프레임 카운터
        real_idx = start_idx + i
        Visualizer.draw_text_with_bg(
            frame, 
            f"Frame: {real_idx}/{len(frame_files)} (Seg: {i}/{len(target_frames)})", 
            (20, h - 20), 
            align_bottom=True
        )

        out.write(frame)

    out.release()
    print(f"✅ 완료: {save_path}")

# ==============================================================================
# 🎥 4. 각 모드별 구현 (Wrapper Functions)
# ==============================================================================
def generate_17kpt_skeleton_video(frame_dir, kpt_dir, output_path, start_idx=0, end_idx=None, conf_threshold=0.0):
    """17 Keypoints (COCO) 비디오 생성"""
    
    def drawer(frame, data):
        for inst in data.get('instance_info', []):
            if inst.get('score', 1.0) <= conf_threshold: continue
            kpts = np.array(inst.get('keypoints', []))
            if kpts.shape[1] < 2: continue
            
            coords = kpts[:, :2].astype(int)
            # 점수가 별도 필드(keypoint_scores)에 있거나 kpts의 3번째 값으로 있을 경우 처리
            if 'keypoint_scores' in inst:
                scores = np.array(inst['keypoint_scores'])
            elif kpts.shape[1] > 2:
                scores = kpts[:, 2]
            else:
                scores = np.ones(len(coords))

            obj_id = inst.get('instance_id', inst.get('id', '?'))

            # 1. Lines
            for u, v in Config.LINKS_17:
                if u < len(coords) and v < len(coords):
                    if scores[u] > conf_threshold and scores[v] > conf_threshold:
                        if coords[u][0] > 0 and coords[v][0] > 0:
                            cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)
            
            # 2. Dots
            for idx, (x, y) in enumerate(coords):
                if not (5 <= idx <= 16): continue
                if scores[idx] > conf_threshold and x > 0:
                    color = Config.COLOR_RIGHT if idx in Config.KPT_17_RIGHT else \
                            (Config.COLOR_LEFT if idx in Config.KPT_17_LEFT else (0, 255, 0))
                    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
            
            # 3. BBox & ID
            bbox = inst.get('bbox')
            
            # bbox가 없는 경우 키포인트 기반 자동 계산 (Fallback)
            if bbox is None:
                valid_mask = (scores > conf_threshold) & (coords[:, 0] > 0) & (coords[:, 1] > 0)
                if np.any(valid_mask):
                    v_coords = coords[valid_mask]
                    min_x, min_y = np.min(v_coords, axis=0)
                    max_x, max_y = np.max(v_coords, axis=0)
                    pad = 10
                    h_img, w_img = frame.shape[:2]
                    bbox = [max(0, min_x - pad), max(0, min_y - pad), min(w_img, max_x + pad), min(h_img, max_y + pad)]

            # 그리기 함수 호출 (이제 이중 리스트도 처리됨)
            if bbox:
                Visualizer.draw_bbox_and_id(frame, bbox, obj_id, Config.COLOR_ID)
                
        return frame

    create_video_engine(frame_dir, output_path, kpt_dir, drawer, start_idx=start_idx, end_idx=end_idx)

def generate_133kpt_skeleton_video(frame_dir, kpt_dir, output_path, start_idx=0, end_idx=None, conf_threshold=0.05):
    """133 Keypoints (WholeBody) 비디오 생성"""
    
    def drawer(frame, data):
        for inst in data.get('instance_info', []):
            if inst.get('score', 1.0) <= 0.0: continue
            kpts = np.array(inst.get('keypoints', []))
            if kpts.shape[1] < 2: continue
            
            coords = kpts[:, :2].astype(int)
            scores = kpts[:, 2] if kpts.shape[1] > 2 else np.ones(len(coords))
            obj_id = inst.get('instance_id', inst.get('id', '?'))

            # Lines
            for part, links in Config.LINKS_133.items():
                # 색상 매핑
                color = Config.COLOR_BODY
                if part == 'face': color = Config.COLOR_FACE
                elif part == 'l_hand': color = Config.COLOR_HAND_L
                elif part == 'r_hand': color = Config.COLOR_HAND_R
                elif part == 'feet': color = Config.COLOR_FOOT_L 

                for u, v in links:
                    if u < len(coords) and v < len(coords):
                        if scores[u] > conf_threshold and scores[v] > conf_threshold:
                            if coords[u][0] > 0 and coords[v][0] > 0:
                                cv2.line(frame, tuple(coords[u]), tuple(coords[v]), color, 1, cv2.LINE_AA)
            
            # Dots
            for idx, (x, y) in enumerate(coords):
                if scores[idx] > conf_threshold and x > 0:
                    cv2.circle(frame, (x, y), 2, Config.COLOR_BODY, -1, cv2.LINE_AA)

            # BBox & ID
            if 'bbox' in inst:
                Visualizer.draw_bbox_and_id(frame, inst['bbox'], obj_id, Config.COLOR_ID)
        return frame

    create_video_engine(frame_dir, output_path, kpt_dir, drawer, start_idx=start_idx, end_idx=end_idx)

def generate_sam_video(frame_dir, json_dir, output_path, start_idx=0, end_idx=None, fps=30.0, alpha=0.5):
    """SAM Segmentation 비디오 생성"""
    _color_cache = {}

    def get_color(oid):
        if oid not in _color_cache:
            _color_cache[oid] = [random.randint(50, 255) for _ in range(3)]
        return _color_cache[oid]

    def drawer(frame, data):
        overlay = frame.copy()
        bboxes_to_draw = []

        if "objects" in data:
            for obj in data["objects"]:
                obj_id = obj.get("id", "?")
                color = get_color(obj_id)
                rle = obj.get("segmentation", {}).get("counts")
                
                if rle:
                    h, w = frame.shape[:2]
                    mask = Visualizer.rle_to_mask(rle, h, w)
                    if mask.sum() > 0:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, color, 2)
                        cv2.fillPoly(overlay, contours, color)
                        
                        ys, xs = np.where(mask)
                        if len(ys) > 0:
                            bbox = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
                            bboxes_to_draw.append((bbox, obj_id, color))

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        for bbox, oid, col in bboxes_to_draw:
            Visualizer.draw_bbox_and_id(frame, bbox, oid, col)
            
        return frame

    create_video_engine(frame_dir, output_path, json_dir, drawer, fps=fps, start_idx=start_idx, end_idx=end_idx)

def generate_filtered_id_skeleton_video(frame_dir, kpt_dir, output_path, target_ids, start_idx=0, end_idx=None, conf_threshold=0.01):
    """특정 ID들만 선택하여 스켈레톤 비디오를 만듭니다."""
    filter_set = {target_ids} if isinstance(target_ids, (int, str)) else set(target_ids) # 필터 세트를 만듭니다.

    def drawer(frame, data):
        for inst in data.get('instance_info', []): # 모든 인스턴스 확인입니다.
            obj_id = inst.get('instance_id', inst.get('id', '?')) # ID 확인입니다.
            if obj_id not in filter_set: continue # ⭐️ 필터링 대상이 아니면 건너뜁니다.
            
            kpts = np.array(inst.get('keypoints', [])) # 좌표 데이터입니다.
            scores = np.array(inst.get('keypoint_scores', np.ones(len(kpts)))) # 신뢰도입니다.
            coords = kpts[:, :2].astype(int)

            for u, v in Config.LINKS_17: # 뼈대 연결입니다.
                if scores[u] > conf_threshold and scores[v] > conf_threshold:
                    cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)
            
            for idx, (x, y) in enumerate(coords): # 관절점 그리기입니다.
                if idx in Config.KPT_17_LEFT or idx in Config.KPT_17_RIGHT:
                    color = Config.COLOR_RIGHT if idx in Config.KPT_17_RIGHT else Config.COLOR_LEFT
                    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)

            Visualizer.draw_bbox_and_id(frame, inst.get('bbox'), obj_id, Config.COLOR_ID) # Bbox 표시입니다.
        return frame

    create_video_engine(frame_dir, output_path, kpt_dir, drawer, start_idx=start_idx, end_idx=end_idx) # 엔진 실행입니다.
    
def draw_skeleton_on_black(frame_kpt, canvas_size=(512, 512), is_normalized=True):
    """
    제공된 Config 클래스의 색상 및 링크 정보를 사용하여 검은 배경에 스켈레톤을 그립니다.
    """
    # 1. 검은색 배경 캔버스 생성
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    
    kpts = frame_kpt.copy()
    
    # 2. 좌표 스케일링 (정규화된 데이터 대응)
    if is_normalized:
        # 정규화된 좌표(-1~1)를 캔버스 중앙에 맞게 확대 및 이동
        scale = min(canvas_size) * 0.3
        offset = np.array([canvas_size[1] // 2, canvas_size[0] // 2])
        kpts[:, :2] = (kpts[:, :2] * scale) + offset
    
    coords = kpts[:, :2].astype(int)
    scores = kpts[:, 2] if kpts.shape[1] > 2 else np.ones(len(kpts))

    # 3. 뼈대 연결 (Config.LINKS_17 사용)
    for u, v in Config.LINKS_17:
        if scores[u] > 0 and scores[v] > 0:
            pt1 = tuple(coords[u])
            pt2 = tuple(coords[v])
            # Config에 정의된 COLOR_SKELETON (100, 100, 100) 사용
            cv2.line(canvas, pt1, pt2, Config.COLOR_SKELETON, 2, cv2.LINE_AA)

    # 4. 관절점 표시 (좌/우 색상 구분)
    for idx, (x, y) in enumerate(coords):
        if scores[idx] > 0:
            # Config.KPT_17_RIGHT/LEFT 세트를 사용하여 색상 결정
            if idx in Config.KPT_17_RIGHT:
                color = Config.COLOR_RIGHT  # (0, 0, 255) - Red
            elif idx in Config.KPT_17_LEFT:
                color = Config.COLOR_LEFT   # (255, 0, 0) - Blue
            else:
                color = (0, 255, 0) # 기본 녹색
                
            cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)

    return canvas

def create_skeleton_video(data_array, output_path, fps=30, canvas_size=(512, 512), is_normalized=True):
    """
    (Frames, 17, 3) 넘파이 배열을 입력받아 검은 배경 스켈레톤 비디오를 생성합니다.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_size[1], canvas_size[0]))

    for i in tqdm(range(len(data_array)), desc="Rendering Video"):
        # 프레임 그리기
        frame = draw_skeleton_on_black(data_array[i], canvas_size, is_normalized)
        
        # 프레임 번호 및 ID 정보 표시 (선택 사항)
        cv2.putText(frame, f"Frame: {i}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 1, cv2.LINE_AA)
        
        out.write(frame)

    out.release()
    print(f"✅ 비디오 저장 완료: {output_path}")

def generate_counting_skeleton_video(frame_dir, kpt_dir, output_path, counter, patient_id=None, start_idx=0, end_idx=None, conf_threshold=0.0):
    """
    원본 영상에 특정 ID의 스켈레톤(12Kpts)을 오버레이하고, 
    카운팅 결과 및 실시간 측정값(각도/거리)을 우측 하단에 표시합니다.
    """
    def drawer(frame, data):
        target_kpt = np.zeros((12, 3))
        found_target = False
        current_metrics = {'left': None, 'right': None} # 🌟 메트릭 저장을 위한 변수
        
        for inst in data.get('instance_info', []):
            obj_id = inst.get('instance_id', inst.get('id', '?'))
            if patient_id is not None and obj_id != patient_id:
                continue
                
            kpts = np.array(inst.get('keypoints', []))
            if kpts.shape[0] < 17: continue 
            
            full_coords = kpts[:, :2].astype(int)
            full_scores = np.array(inst['keypoint_scores']) if 'keypoint_scores' in inst else \
                          (kpts[:, 2] if kpts.shape[1] > 2 else np.ones(len(full_coords)))
            
            coords = full_coords[5:17]
            scores = full_scores[5:17]
            target_kpt = np.hstack([coords, scores.reshape(-1, 1)])
            found_target = True

            # 스켈레톤 라인 그리기
            for u, v in Config.LINKS_12:
                if u < len(coords) and v < len(coords) and scores[u] > conf_threshold and scores[v] > conf_threshold:
                    if coords[u][0] > 0 and coords[v][0] > 0:
                        cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)
            
            # 관절 점 그리기
            for idx, (x, y) in enumerate(coords):
                if scores[idx] > conf_threshold and x > 0:
                    color = Config.COLOR_RIGHT if idx % 2 != 0 else Config.COLOR_LEFT
                    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                    
            Visualizer.draw_bbox_and_id(frame, inst.get('bbox'), obj_id, Config.COLOR_ID)
            break
            
        # 1. 카운트 업데이트 및 메트릭 추출
        if found_target and np.any(target_kpt):
            norm_kpt = normalize_skeleton_array(np.expand_dims(target_kpt, axis=0))[0]
            # 🌟 process_frame에서 반환된 metrics를 저장합니다.
            current_metrics, _ = counter.process_frame(norm_kpt)
            
        # --- UI 렌더링 영역 ---
        h_img, w_img = frame.shape[:2]
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        
        # 2. [우측 상단] 카운트 표시 (기존 로직)
        l_count, r_count = counter.counts.get('left', 0), counter.counts.get('right', 0)
        text_l, text_r = f"Left: {l_count}", f"Right: {r_count}"
        (tw_l, th_l), _ = cv2.getTextSize(text_l, font, scale, thickness)
        (tw_r, th_r), _ = cv2.getTextSize(text_r, font, scale, thickness)
        
        x_l, x_r = w_img - tw_l - 30, w_img - tw_r - 30
        pad = 12
        cv2.rectangle(frame, (x_l - pad, 60 - th_l - pad), (x_l + tw_l + pad, 60 + pad), (220, 50, 50), -1) 
        cv2.rectangle(frame, (x_r - pad, 130 - th_r - pad), (x_r + tw_r + pad, 130 + pad), (50, 50, 220), -1) 
        cv2.putText(frame, text_l, (x_l, 60), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, text_r, (x_r, 130), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 🌟 3. [우측 하단] 실시간 메트릭(각도/거리) 표시 추가
        # YAML 설정에서 calc_method를 가져와 단위를 표시합니다.
        method_label = "deg" if counter.calc_method == "angle" else "dist"
        
        l_val = current_metrics.get('left')
        r_val = current_metrics.get('right')
        
        # 값이 존재할 경우 소수점 1자리까지 표시, 없을 경우 N/A 표시
        str_l = f"L-{method_label}: {l_val:.1f}" if l_val is not None else f"L-{method_label}: N/A"
        str_r = f"R-{method_label}: {r_val:.1f}" if r_val is not None else f"R-{method_label}: N/A"
        
        # 하단 텍스트는 상단보다 약간 작게 설정
        m_scale, m_thick = 0.8, 2
        (mw_l, mh_l), _ = cv2.getTextSize(str_l, font, m_scale, m_thick)
        (mw_r, mh_r), _ = cv2.getTextSize(str_r, font, m_scale, m_thick)
        
        mx_l, my_l = w_img - mw_l - 30, h_img - 80
        mx_r, my_r = w_img - mw_r - 30, h_img - 30
        
        # 가독성을 위한 검은색 배경 박스
        cv2.rectangle(frame, (mx_l - 10, my_l - mh_l - 10), (mx_l + mw_l + 10, my_l + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (mx_r - 10, my_r - mh_r - 10), (mx_r + mw_r + 10, my_r + 10), (0, 0, 0), -1)
        
        # 수치 텍스트 렌더링 (파스텔 톤 파랑/빨강)
        cv2.putText(frame, str_l, (mx_l, my_l), font, m_scale, (255, 200, 200), m_thick, cv2.LINE_AA)
        cv2.putText(frame, str_r, (mx_r, my_r), font, m_scale, (200, 200, 255), m_thick, cv2.LINE_AA)
        
        return frame

    create_video_engine(frame_dir, output_path, kpt_dir, drawer, start_idx=start_idx, end_idx=end_idx)

def generate_12kpt_skeleton_video_from_np(frame_dir, kpt_np, output_path, fps=30, conf_threshold=0.0):
    """
    (N, 12, 3) 형태의 Numpy 배열을 받아 원본 이미지 프레임 위에 오버레이하여 비디오를 생성합니다.
    
    Args:
        frame_dir (str/Path): 원본 이미지 프레임들이 있는 디렉토리 경로.
        kpt_np (np.ndarray): (프레임수, 12관절, 3(x, y, score)) 형태의 Numpy 배열.
        output_path (str/Path): 저장할 mp4 비디오 파일 경로.
        fps (int): 비디오 프레임 레이트 (기본값 30).
        conf_threshold (float): 이 점수 이하인 관절은 그리지 않음.
    """
    frame_path = Path(frame_dir)
    save_path = Path(output_path)
    
    # 1. 프레임 이미지 리스트 확보
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png")))
    if not frame_files:
        print(f"❌ [Error] 프레임 이미지가 없습니다: {frame_dir}")
        return

    n_frames_img = len(frame_files)
    n_frames_np = kpt_np.shape[0]
    
    # 데이터 길이 불일치 방어 로직
    target_len = min(n_frames_img, n_frames_np)
    if n_frames_img != n_frames_np:
        print(f"⚠️ [경고] 프레임 수({n_frames_img})와 Numpy 배열 길이({n_frames_np})가 다릅니다.")
        print(f"   -> 최소 길이인 {target_len} 프레임까지만 생성합니다.")

    # 2. 비디오 Writer 초기화
    save_path.parent.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(frame_files[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"🎬 12Kpt 비디오 생성 중... -> {save_path.name}")

    # 3. 렌더링 루프
    for i in tqdm(range(target_len), desc="Drawing 12Kpt Skeleton"):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None: continue

        # 현재 프레임의 12관절 데이터 추출 (12, 3)
        current_kpts = kpt_np[i]
        coords = current_kpts[:, :2].astype(int)
        scores = current_kpts[:, 2]

        # ----------------------------------------------------
        # 🎨 스켈레톤 그리기 로직 (12 Keypoints 기준)
        # ----------------------------------------------------
        
        # 1. Lines (뼈대 연결)
        for u, v in Config.LINKS_12:
            if u < 12 and v < 12:  # 인덱스 안전 체크
                if scores[u] > conf_threshold and scores[v] > conf_threshold:
                    if coords[u][0] > 0 and coords[v][0] > 0:
                        cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)
        
        # 2. Dots (관절점)
        for idx, (x, y) in enumerate(coords):
            if scores[idx] > conf_threshold and x > 0 and y > 0:
                # 12포인트 기준 색상 매핑
                color = Config.COLOR_RIGHT if idx in Config.KPT_12_RIGHT else Config.COLOR_LEFT
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)

        # 3. 자동 BBox 생성 및 그리기
        valid_mask = (scores > conf_threshold) & (coords[:, 0] > 0) & (coords[:, 1] > 0)
        if np.any(valid_mask):
            v_coords = coords[valid_mask]
            min_x, min_y = np.min(v_coords, axis=0)
            max_x, max_y = np.max(v_coords, axis=0)
            pad = 15 # 박스 여백
            bbox = [max(0, min_x - pad), max(0, min_y - pad), min(w, max_x + pad), min(h, max_y + pad)]
            
            # ID는 Numpy 배열에서 알 수 없으므로 'Filtered' 로 고정 표기
            Visualizer.draw_bbox_and_id(frame, bbox, "Filtered", Config.COLOR_ID)

        # 공통 UI: 프레임 정보 표시
        Visualizer.draw_text_with_bg(
            frame, 
            f"Frame: {i}/{target_len}", 
            (20, h - 20), 
            align_bottom=True
        )

        out.write(frame)

    out.release()
    print(f"✅ 완료: {save_path}")