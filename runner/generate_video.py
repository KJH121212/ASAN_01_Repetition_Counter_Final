from pathlib import Path
import pandas as pd

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/metadata_v2.1.csv")

import sys
sys.path.append(str(BASE_DIR))
from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video, generate_sam_video, generate_133kpt_skeleton_video
from utils.path_list import path_list

df=pd.read_csv(CSV_PATH)
target = 44
common_path = df.loc[target,"common_path"]

paths = path_list(common_path, create_dirs=True)

sam_out_path = paths["test"] / "sam.mp4"


# print(common_path)

generate_17kpt_skeleton_video(
    frame_dir=paths['frame'],
    kpt_dir=paths['keypoint'],
    output_path=paths['mp4'],
    conf_threshold=0
)

# generate_133kpt_skeleton_video(
#     frame_dir=frame_path,
#     kpt_dir=kpt_path,
#     output_path=out_path,
#     conf_threshold=0
# )

# generate_sam_video(
#     frame_dir=paths['frame'],
#     json_dir=paths['sam'],
#     output_path=sam_out_path,
#     fps=30.0,  # FPS 설정
#     alpha=0.5
# )
