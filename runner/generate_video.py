import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from utils.path_list import path_list
from utils.generate_skeleton_video_v2 import generate_integrated_video

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data_v2.1.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================
print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSANJIN_PATH.name})")
meta_df = pd.read_csv(METADATA_PATH) # 전체 메타데이터를 로드합니다.
bosan_df = pd.read_csv(BOSANJIN_PATH) # Bosanjin 세그먼트 데이터를 로드합니다.

target = 1
common_path = bosan_df.iloc[target]['common_path']
paths=path_list(common_path=common_path)
start_idx = bosan_df.iloc[target]['start_frame']
end_idx = bosan_df.iloc[target]['end_frame']

generate_integrated_video(
    frame_dir=paths['frame'],
    output_path="./img/test.mp4",
    skeleton_dir=paths['interp_data'],
    sam_dir=paths['sam'],
    start_idx=start_idx,
    end_idx=end_idx
)