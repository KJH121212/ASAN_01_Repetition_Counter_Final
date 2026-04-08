import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path: str, frame_dir: str, target_short: int = 720, jpeg_quality: int = 80) -> int:
    """영상에서 프레임 추출 (1만 개 단위 하위 폴더 분산 저장)"""
    video_path, frame_dir = Path(video_path), Path(frame_dir)
    
    # 1. 기존 폴더 초기화
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    # 영상 정보 획득
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 리사이즈 스케일 계산 (720p 기준)
    scale = target_short / w if w <= h else target_short / h
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # 💡 폴더 분산 설정
    MAX_PER_FOLDER = 10000 
    count = 0

    for idx in tqdm(range(n_frames), total=n_frames, desc="Extracting Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # 💡 [핵심 수정] 저장될 하위 폴더 번호 계산 및 생성
        # idx가 0~9999면 '00', 10000~19999면 '01' 폴더에 저장됨
        subdir_num = f"{idx // MAX_PER_FOLDER:02d}"
        target_subdir = frame_dir / subdir_num
        target_subdir.mkdir(exist_ok=True) # 폴더가 없으면 생성

        # 프레임 리사이즈
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 💡 [핵심 수정] 최종 저장 경로 (예: frame_dir/00/000123.jpg)
        out_path = target_subdir / f"{idx:06d}.jpg"
        
        cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        count += 1

    cap.release()
    return count