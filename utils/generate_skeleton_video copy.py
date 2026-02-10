import json
import cv2
import random
import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Union

# ==============================================================================
# 🛠️ 1. 설정 및 내부 유틸리티 (Common Settings & Utils)
# ==============================================================================

# BGR 색상 정의
COLOR_SKELETON = (100, 100, 100) # 뼈대: 회색
COLOR_RIGHT = (0, 0, 255)        # 오른쪽: Red
COLOR_LEFT = (255, 0, 0)         # 왼쪽: Blue
COLOR_ID = (0, 255, 0)           # ID: Green
COLOR_TEXT = (255, 255, 255)     # 텍스트: White
COLOR_BODY = (255, 100, 0)       # 133-KPT 몸통
COLOR_FACE = (0, 255, 255)       # 133-KPT 얼굴
COLOR_L_HAND = (0, 0, 255)       # 133-KPT 왼손
COLOR_R_HAND = (255, 0, 0)       # 133-KPT 오른손

# 17 Keypoints 연결 정보
SKELETON_LINKS_17 = [
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), 
    (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
]
LEFT_INDICES = {5, 7, 9, 11, 13, 15}
RIGHT_INDICES = {6, 8, 10, 12, 14, 16}

# 133 Keypoints (WholeBody) 연결 정보
SKELETON_LINKS_133 = {
    'body': [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
    'feet': [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)],
    'face': [(23, 24), (24, 25), (26, 27), (27, 28), (62, 63), (63, 64), (64, 65), (65, 66)],
    'l_hand': [(9, 91), (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98), (98, 99), (91, 100), (100, 101), (101, 102), (102, 103), (91, 104), (104, 105), (105, 106), (106, 107), (91, 108), (108, 109), (109, 110), (110, 111)],
    'r_hand': [(10, 112), (112, 113), (113, 114), (114, 115), (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123), (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131), (131, 132)]
}

_color_map: Dict[int, List[int]] = {}

def get_color(obj_id: int) -> List[int]:
    """객체 ID별 고유 랜덤 색상 반환"""
    if obj_id not in _color_map:
        _color_map[obj_id] = [random.randint(50, 255) for _ in range(3)] # RGB 랜덤 생성
    return _color_map[obj_id]; # 생성된 색상 리스트 반환

def rle_to_mask(rle: List[int], height: int, width: int) -> np.ndarray:
    """RLE 데이터를 이진 마스크로 변환"""
    mask = np.zeros(height * width, dtype=np.uint8); # 1차원 빈 배열 생성
    if not rle: return mask.reshape((height, width)); # RLE가 없으면 빈 마스크 반환
    rle = np.array(rle); # 계산을 위한 넘파이 변환
    starts = rle[0::2] - 1; # 시작점 추출
    lengths = rle[1::2]; # 길이 추출
    for lo, hi in zip(starts, starts + lengths):
        mask[max(lo, 0):min(hi, len(mask))] = 1; # 해당 구간을 1로 채움
    return mask.reshape((height, width)); # 2차원 이미지 형태로 복원

# ==============================================================================
# 🎨 2. 시각화 헬퍼 함수 (Drawing Helpers)
# ==============================================================================

def draw_frame_counter(frame: np.ndarray, current_idx: int, total_frames: int):
    """좌측 하단에 프레임 번호 표시"""
    text = f"Frame: {current_idx}/{total_frames}";          # 텍스트 생성
    font = cv2.FONT_HERSHEY_SIMPLEX;                        # 폰트 설정
    (w, h), _ = cv2.getTextSize(text, font, 0.6, 1);        # 텍스트 사이즈 계산
    img_h = frame.shape[0];                                 # 이미지 높이 획득
    cv2.rectangle(frame, (20, img_h - 45), (20 + w + 20, img_h - 15), (0, 0, 0), -1);   # 검은 배경 박스
    cv2.putText(frame, text, (30, img_h - 25), font, 0.6, COLOR_TEXT, 1, cv2.LINE_AA);  # 흰색 글씨 작성

def draw_bbox_and_id(frame: np.ndarray, mask: np.ndarray, obj_id: int):
    """프레임에 BBox와 ID 라벨 그리기"""
    color = get_color(obj_id);                                                      # ID에 따른 색상 획득
    y, x = np.where(mask);                                                          # 마스크 영역의 좌표 획득
    if len(y) > 0:
        x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y);                # BBox 좌표 계산
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2);                         # BBox 사각형 그리기
        label = f"ID: {obj_id}"; # ID 라벨 텍스트
        (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1);   # 텍스트 크기 측정
        cv2.rectangle(frame, (x1, y1 - h_t - 10), (x1 + w_t + 10, y1), color, -1);  # 라벨 배경 박스
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); # 라벨 텍스트 작성

# ==============================================================================
# 🚀 3. 메인 시각화 비디오 생성 함수들 (Main Logic)
# ==============================================================================

def generate_17kpt_skeleton_video(frame_dir: str, kpt_dir: str, output_path: str, conf_threshold: float = 0.0):
    """17 Keypoints 표준 스켈레톤 비디오 생성"""
    frame_path, json_path, save_path = Path(frame_dir), Path(kpt_dir), Path(output_path);   # 경로 객체화
    save_path.parent.mkdir(parents=True, exist_ok=True);                                    # 저장 디렉토리 생성
    frame_files = sorted(list(frame_path.glob("*.jpg")) + list(frame_path.glob("*.png")));  # 이미지 로드
    json_files = sorted(list(json_path.glob("*.json")));                                    # JSON 로드
    
    first_frame = cv2.imread(str(frame_files[0]));                                      # 첫 프레임 읽기
    h, w = first_frame.shape[:2];                                                       # 해상도 정보 획득
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h)); # 비디오 작성기 초기화

    for i in tqdm(range(len(frame_files)), desc="Rendering 17-KPT"):
        frame = cv2.imread(str(frame_files[i]));                                                # 프레임 이미지 읽기
        draw_frame_counter(frame, i, len(frame_files));                                         # 프레임 번호 표시
        
        if i < len(json_files):
            with open(json_files[i], 'r') as f: data = json.load(f);                            # JSON 파일 열기
            for inst in data.get('instance_info', []):
                coords = np.array(inst['keypoints']);                                           # 좌표 데이터 추출
                scores = np.array(inst.get('keypoint_scores', np.ones(len(coords))));           # 점수 데이터 추출
                
                for u, v in SKELETON_LINKS_17: # 뼈대 연결 루프
                    if scores[u] > conf_threshold and scores[v] > conf_threshold:
                        pt1, pt2 = tuple(coords[u][:2].astype(int)), tuple(coords[v][:2].astype(int));  # 좌표 정수화
                        cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2, cv2.LINE_AA);                      # 뼈대 그리기
                
                for idx, (x, y) in enumerate(coords[:, :2]):                                        # 관절 포인트 루프
                    if 5 <= idx <= 16 and scores[idx] > conf_threshold:
                        color = COLOR_RIGHT if idx in RIGHT_INDICES else COLOR_LEFT;                    # 좌우 색상 결정
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA);                 # 관절 포인트 그리기
        out.write(frame);   # 프레임 저장
    out.release();          # 리소스 해제

def generate_sam_video(frame_dir: str, json_dir: str, output_path: str, alpha: float = 0.5):
    """SAM Segmentation 결과 오버레이 비디오 생성"""
    frame_dir, json_dir, output_path = Path(frame_dir), Path(json_dir), Path(output_path);  # 경로 설정
    frame_files = sorted(glob.glob(str(frame_dir / "*.jpg")));                              # 이미지 파일 목록
    first_img = cv2.imread(frame_files[0]);                                                 # 첫 이미지 로드
    h, w = first_img.shape[:2];                                                             # 크기 획득
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h));   # 저장 설정

    for i, img_path in enumerate(tqdm(frame_files, desc="Rendering SAM")):
        frame = cv2.imread(img_path);                                       # 원본 프레임 로드
        overlay = frame.copy();                                             # 마스크를 그릴 복사본 생성
        json_path = json_dir / f"{i:06d}.json";                             # 파일명 매칭
        objs_to_draw = [];                                                  # ID 표시를 위한 임시 저장 리스트
        
        if json_path.exists():
            with open(json_path, 'r') as f: data = json.load(f);        # JSON 로드
            for obj in data.get("objects", []):
                rle = obj.get("segmentation", {}).get("counts");        # RLE 데이터 획득
                if rle:
                    mask = rle_to_mask(rle, h, w);                      # 마스크 변환
                    color = get_color(obj.get("id"));                   # 고유 색상 획득
                    overlay[mask == 1] = color; # 오버레이 레이어에 색칠
                    objs_to_draw.append((mask, obj.get("id"))); # 후처리를 위해 저장
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame); # 프레임 합성 (반투명)
        for mask, obj_id in objs_to_draw: draw_bbox_and_id(frame, mask, obj_id); # BBox와 ID 그리기
        draw_frame_counter(frame, i, len(frame_files)); # 프레임 카운터 표시
        out.write(frame); # 비디오 쓰기
    out.release(); # 종료

def generate_133kpt_skeleton_video(frame_dir: str, kpt_dir: str, output_path: str, conf_threshold: float = 0.05):
    """133 Keypoints (WholeBody) 비디오 생성"""
    frame_path, json_path, save_path = Path(frame_dir), Path(kpt_dir), Path(output_path); # 경로 초기화
    frame_files = sorted(list(frame_path.glob("*.jpg"))); # 파일 리스트업
    first_frame = cv2.imread(str(frame_files[0])); # 첫 프레임 로드
    h, w = first_frame.shape[:2]; # 크기 확인
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h)); # 저장기 설정

    for i in tqdm(range(len(frame_files)), desc="Rendering 133-KPT"):
        frame = cv2.imread(str(frame_files[i])); # 이미지 로드
        json_file = json_path / f"{Path(frame_files[i]).stem}.json"; # JSON 파일 매칭
        if json_file.exists():
            with open(json_file, 'r') as f: data = json.load(f); # 데이터 로드
            for inst in data.get('instance_info', []):
                raw_kpts = np.array(inst['keypoints']); # (133, 3) 또는 (133, 2)
                coords = raw_kpts[:, :2]; # 좌표 슬라이싱
                scores = raw_kpts[:, 2] if raw_kpts.shape[1] > 2 else np.ones(len(coords)); # 점수 슬라이싱
                
                for part, links in SKELETON_LINKS_133.items(): # 부위별 그리기
                    color = {'body': COLOR_BODY, 'face': COLOR_FACE, 'l_hand': COLOR_L_HAND, 'r_hand': COLOR_R_HAND}.get(part, COLOR_BODY); # 색상 설정
                    for u, v in links:
                        if u < len(coords) and v < len(coords) and scores[u] > conf_threshold and scores[v] > conf_threshold:
                            cv2.line(frame, tuple(coords[u].astype(int)), tuple(coords[v].astype(int)), color, 1, cv2.LINE_AA); # 뼈대 선 그리기
        out.write(frame); # 프레임 저장
    out.release(); # 리소스 해제