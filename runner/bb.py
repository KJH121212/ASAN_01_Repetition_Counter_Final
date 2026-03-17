import sys
import pandas as pd
from pathlib import Path

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from utils.path_list import path_list
from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video, generate_12kpt_skeleton_video_from_np
from utils.extract_kpt import normalize_skeleton_array, extract_id_keypoints, save_12kpt_to_17kpt_json
from utils.parser import parse_common_path

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================
print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSANJIN_PATH.name})")
meta_df = pd.read_csv(METADATA_PATH)
bosan_df = pd.read_csv(BOSANJIN_PATH)

target = 15
common_path = bosan_df.iloc[target]['common_path']
start_frame = bosan_df.iloc[target]['start_frame']
end_frame = bosan_df.iloc[target]['end_frame']

paths = path_list(common_path)
patient_id = int(meta_df.loc[meta_df['common_path'] == common_path, 'patient_id'].iloc[0]) 

from ground_truth_pipeline.step4_assign_ids import assign_sam_ids_to_keypoints

assign_sam_ids_to_keypoints(
    sam_dir=paths['sam'],
    kpt_dir=paths['keypoint'],
    output_dir=paths['test']/"assign_v1.4"
)

generate_17kpt_skeleton_video(
    frame_dir=paths['frame'],
    kpt_dir=paths['test']/"assign_v1.4",
    output_path=paths['test']/"assign_v1.4.mp4",
    conf_threshold=0
)


