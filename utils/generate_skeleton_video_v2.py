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

def generate_integrated_video(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    skeleton_dir: Optional[Union[str, Path]] = None,
    sam_dir: Optional[Union[str, Path]] = None,
    target_ids: Optional[List[Union[int, str]]] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    conf_threshold: float = 0.0,
    fps: int = 30
):
    """
    프레임 이미지에 스켈레톤(17kpt)과 SAM 마스크를 통합하여 오버레이 비디오를 생성합니다.
    데이터가 없는 경우(None) 해당 요소는 건너뛰고 렌더링합니다.
    """
    _sam_color_cache = {} # SAM 마스크 ID별 색상 고정을 위한 캐시입니다.
    
    def get_sam_color(oid):
        if oid not in _sam_color_cache:
            _sam_color_cache[oid] = [random.randint(50, 255) for _ in range(3)]
        return _sam_color_cache[oid]

    def integrated_drawer(frame, data_dict):
        h_img, w_img = frame.shape[:2] # 프레임 크기 정보입니다.

        # 1. SAM 데이터 처리 (하단 배치 및 흰색 BBox)
        sam_data = data_dict.get('sam', {})
        if "objects" in sam_data:
            overlay = frame.copy()
            for obj in sam_data["objects"]:
                obj_id = obj.get("id", "?")
                if target_ids and obj_id not in target_ids: continue
                
                color = get_sam_color(obj_id)
                rle = obj.get("segmentation", {}).get("counts")
                
                if rle:
                    mask = Visualizer.rle_to_mask(rle, h_img, w_img)
                    if mask.sum() > 0:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, color, 2)
                        cv2.fillPoly(overlay, contours, color)
                        
                        # SAM 전용 BBox 계산 (하단에 ID 표시)
                        ys, xs = np.where(mask)
                        xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2) # 하얀색 BBox입니다.
                        # 하단에 ID 표시 (ymax 근처)
                        Visualizer.draw_text_with_bg(frame, f"SAM ID: {obj_id}", (xmin, ymax + 20), bg_color=(255, 255, 255), txt_color=(0, 0, 0))
            
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 2. Skeleton 데이터 처리 (상단 배치 및 검은색 BBox)
        skel_data = data_dict.get('skeleton', {})
        for inst in skel_data.get('instance_info', []):
            obj_id = inst.get('instance_id', inst.get('id', '?'))
            if target_ids and obj_id not in target_ids: continue
            if inst.get('score', 1.0) < conf_threshold: continue

            kpts = np.array(inst.get('keypoints', []))
            if kpts.shape[0] == 0: continue

            coords = kpts[:, :2].astype(int)
            scores = inst.get('keypoint_scores', kpts[:, 2] if kpts.shape[1] > 2 else np.ones(len(coords)))

            # 뼈대 및 관절점 렌더링
            for u, v in Config.LINKS_17:
                if u < len(coords) and v < len(coords) and scores[u] > conf_threshold and scores[v] > conf_threshold:
                    cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)
            for idx, (x, y) in enumerate(coords):
                if scores[idx] > conf_threshold and x > 0:
                    dot_color = Config.COLOR_RIGHT if idx in Config.KPT_17_RIGHT else (Config.COLOR_LEFT if idx in Config.KPT_17_LEFT else (0, 255, 0))
                    cv2.circle(frame, (x, y), 4, dot_color, -1, cv2.LINE_AA)

            # Skeleton 전용 BBox (상단에 ID 표시, 검은색)
            bbox = inst.get('bbox')
            if bbox:
                bbox_flat = np.array(bbox).flatten().astype(int)
                xmin, ymin, xmax, ymax = bbox_flat[:4]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2) # 검은색 BBox입니다.
                # 상단에 ID 표시 (ymin 근처)
                Visualizer.draw_text_with_bg(frame, f"SKEL ID: {obj_id}", (xmin, ymin - 10), bg_color=(0, 0, 0), txt_color=(255, 255, 255))

        return frame

    # --- 엔진 실행부 ---
    frame_path = Path(frame_dir) # 경로 객체화입니다.
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png"))) # 파일 목록을 정렬합니다.
    
    # 구간 슬라이싱
    final_end = end_idx if end_idx is not None else len(frame_files) # 종료 인덱스 설정입니다.
    target_frames = frame_files[start_idx:final_end] # 작업 대상 프레임들입니다.
    
    if not target_frames: return # 대상이 없으면 종료합니다.

    # 비디오 작성기 초기화
    Path(output_path).parent.mkdir(parents=True, exist_ok=True) # 폴더가 없으면 생성합니다.
    first_img = cv2.imread(str(target_frames[0])) # 첫 프레임으로 크기를 결정합니다.
    h, w = first_img.shape[:2] # 해상도 정보입니다.
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) # Writer 객체 생성입니다.

    for i, f_file in enumerate(tqdm(target_frames, desc="Integrated Rendering")):
        frame = cv2.imread(str(f_file)) # 프레임 이미지를 읽습니다.
        combined_data = {'skeleton': {}, 'sam': {}} # 현재 프레임의 데이터를 담을 사전입니다.

        # 스켈레톤 JSON 로드 시도
        if skeleton_dir:
            skel_json = Path(skeleton_dir) / f"{f_file.stem}.json"
            if skel_json.exists():
                try:
                    with open(skel_json, 'r') as f:
                        combined_data['skeleton'] = json.load(f) # 정상적인 경우 데이터를 로드합니다.
                except json.JSONDecodeError as e:
                    # 🌟 파일이 깨져 있을 경우 에러를 출력하고 빈 데이터를 할당하여 넘어갑니다.
                    print(f"⚠️ [Warning] 스켈레톤 JSON 손상됨: {skel_json.name} ({e})")
                    combined_data['skeleton'] = {} 

        # SAM JSON 로드 시도
        if sam_dir:
            sam_json = Path(sam_dir) / f"{f_file.stem}.json"
            if sam_json.exists():
                try:
                    with open(sam_json, 'r') as f:
                        combined_data['sam'] = json.load(f) # 정상적인 경우 데이터를 로드합니다.
                except json.JSONDecodeError as e:
                    # 🌟 SAM 파일도 마찬가지로 예외 처리를 해줍니다.
                    print(f"⚠️ [Warning] SAM JSON 손상됨: {sam_json.name} ({e})")
                    combined_data['sam'] = {}

        # 통합 드로잉 콜백 실행
        frame = integrated_drawer(frame, combined_data) # 모든 요소를 프레임에 입힙니다.
        
        # 하단 정보 텍스트 (옵션)
        info = f"Frame: {start_idx + i} | ID Filter: {target_ids if target_ids else 'ALL'}" # 현재 상태 정보입니다.
        Visualizer.draw_text_with_bg(frame, info, (20, h - 20), align_bottom=True) # 화면에 표시합니다.

        out.write(frame) # 비디오 파일에 프레임을 저장합니다.

    out.release() # 비디오 파일을 닫습니다.
    print(f"✅ 통합 비디오 생성 완료: {output_path}") # 최종 완료 메시지입니다.

def generate_skeleton_video_np(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    skeleton_np: Optional[np.ndarray] = None, # (Frames, K, 3) 형태의 넘파이 배열입니다.
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    conf_threshold: float = 0.0,
    fps: int = 30
):
    """
    Numpy 스켈레톤 데이터를 프레임 이미지 위에 그려 비디오를 생성합니다.
    skeleton_np: (N, 12, 3) 또는 (N, 17, 3) 형태를 지원합니다.
    """

    def integrated_drawer(frame, skel_frame_data): # 🌟 sam_data 인자를 제거했습니다.
        h_img, w_img = frame.shape[:2] 

        if skel_frame_data is not None and np.any(skel_frame_data):
            num_kpts = skel_frame_data.shape[0]
            coords = skel_frame_data[:, :2].astype(int)
            scores = skel_frame_data[:, 2] if skel_frame_data.shape[1] > 2 else np.ones(num_kpts)

            # 포인트 수에 따른 설정 분기
            if num_kpts == 12:
                links = Config.LINKS_12
                left_idx, right_idx = Config.KPT_12_LEFT, Config.KPT_12_RIGHT
            else:
                links = Config.LINKS_17
                left_idx, right_idx = Config.KPT_17_LEFT, Config.KPT_17_RIGHT

            # 1. 뼈대 그리기
            for u, v in links:
                if u < num_kpts and v < num_kpts:
                    if scores[u] > conf_threshold and scores[v] > conf_threshold:
                        if coords[u][0] > 0 and coords[v][0] > 0:
                            cv2.line(frame, tuple(coords[u]), tuple(coords[v]), Config.COLOR_SKELETON, 2, cv2.LINE_AA)

            # 2. 관절점 그리기
            for idx, (x, y) in enumerate(coords):
                if scores[idx] > conf_threshold and x > 0:
                    dot_color = Config.COLOR_RIGHT if idx in right_idx else (Config.COLOR_LEFT if idx in left_idx else (0, 255, 0))
                    cv2.circle(frame, (x, y), 4, dot_color, -1, cv2.LINE_AA)

            # 3. 상단 BBox (검은색) - 자동 계산
            valid_mask = (scores > conf_threshold) & (coords[:, 0] > 0)
            if np.any(valid_mask):
                v_coords = coords[valid_mask]
                xmin, ymin = np.min(v_coords, axis=0)
                xmax, ymax = np.max(v_coords, axis=0)
                # 약간의 여백(15px)을 추가하여 그립니다.
                cv2.rectangle(frame, (int(xmin-15), int(ymin-15)), (int(xmax+15), int(ymax+15)), (0, 0, 0), 2)
                Visualizer.draw_text_with_bg(frame, "SKEL NP", (int(xmin-15), int(ymin-25)), bg_color=(0, 0, 0), txt_color=(255, 255, 255))

        return frame

    # --- 엔진 실행부 ---
    frame_path = Path(frame_dir)
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png")))
    
    final_end = end_idx if end_idx is not None else len(frame_files)
    target_frames = frame_files[start_idx:final_end]
    
    if not target_frames: return

    # 데이터 길이 정렬
    num_to_process = len(target_frames)
    if skeleton_np is not None:
        num_to_process = min(num_to_process, len(skeleton_np))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(target_frames[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in tqdm(range(num_to_process), desc="Rendering Skeleton Video"):
        frame = cv2.imread(str(target_frames[i]))
        if frame is None: continue

        current_skel = skeleton_np[i] if skeleton_np is not None else None

        # 🌟 인자 개수를 맞추어 호출합니다.
        frame = integrated_drawer(frame, current_skel)
        
        info_text = f"Frame: {start_idx + i} | Kpts: {current_skel.shape[0] if current_skel is not None else 0}"
        Visualizer.draw_text_with_bg(frame, info_text, (20, h - 20), align_bottom=True)

        out.write(frame)

    out.release()
    print(f"✅ 비디오 저장 완료: {output_path}")