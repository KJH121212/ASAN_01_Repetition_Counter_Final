import sys
import pandas as pd
from pathlib import Path
# tqdm 제거

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
# 프로젝트 루트 경로 (사용자 환경에 맞게 수정)
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from utils.path_list import path_list
from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video, generate_12kpt_skeleton_video_from_np, generate_filtered_id_skeleton_video
from utils.extract_kpt import normalize_skeleton_array, extract_id_keypoints
from utils.parser import parse_common_path

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================

meta_df = pd.read_csv(METADATA_PATH) # 메타데이터 CSV 파일을 읽어 데이터프레임으로 불러옵니다.
bosan_df = pd.read_csv(BOSANJIN_PATH) # 보산진 세그먼트 데이터 CSV 파일을 읽어 데이터프레임으로 불러옵니다.
