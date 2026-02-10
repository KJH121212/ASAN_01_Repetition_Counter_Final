import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/metadata_v2.0.csv")

import sys
sys.path.append(str(BASE_DIR))
from utils.generate_skeleton_video import generate_17kpt_skeleton_video, generate_sam_video, generate_133kpt_skeleton_video

df=pd.read_csv(CSV_PATH)

target = 29
common_path = df.loc[target,"common_path"]
frame_path = DATA_DIR / "1_FRAME" / common_path
kpt_path = DATA_DIR / "2_KEYPOINTS" / common_path
sam_path = DATA_DIR / "8_SAM" / common_path
out_path = DATA_DIR / "3_MP4" / f"{common_path}.mp4"

frame_path = DATA_DIR / "walking_data/FRAME/lateral__walking__1"
kpt_path = DATA_DIR / "walking_data/sapiens_133kpt"
out_path = DATA_DIR / "walking_data" / f"133_v5.1.mp4"

# generate_17kpt_skeleton_video(
#     frame_dir=frame_path,
#     kpt_dir=kpt_path,
#     output_path=out_path,
#     conf_threshold=0
# )

generate_133kpt_skeleton_video(
    frame_dir=frame_path,
    kpt_dir=kpt_path,
    output_path=out_path,
    conf_threshold=0
)

# generate_sam_video(
#     frame_dir=frame_path,
#     json_dir=sam_path,
#     output_path=out_path,
#     fps=30.0,  # FPS 설정
#     alpha=0.5
# )
