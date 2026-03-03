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
from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video, generate_12kpt_skeleton_video_from_np
from utils.extract_kpt import normalize_skeleton_array, extract_id_keypoints, save_12kpt_to_17kpt_json
from utils.parser import parse_common_path

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================

meta_df = pd.read_csv(METADATA_PATH) # 메타데이터 CSV 파일을 읽어 데이터프레임으로 불러옵니다.
bosan_df = pd.read_csv(BOSANJIN_PATH) # 보산진 세그먼트 데이터 CSV 파일을 읽어 데이터프레임으로 불러옵니다.

target = 91
paths = path_list(meta_df.iloc[target]['common_path'])

from ground_truth_pipeline.step4_assign_ids import assign_sam_ids_to_keypoints
from utils.postprocessing import apply_custom_kalman

kpt = extract_id_keypoints(
    json_dir=paths['interp_data'],
    target_id=meta_df.iloc[target]['patient_id']
)

kpt = apply_custom_kalman(kpt, threshold=50.0, q_std=1.0, r_std=5.0)

save_12kpt_to_17kpt_json(
    src_dir=paths['interp_data'],
    output_dir=paths['interp_data'],
    kpt_array=kpt,
    target_id=meta_df.iloc[target]['patient_id']
)



# from ground_truth_pipeline.step2_extract_poses import extract_keypoints

# extract_keypoints(
#     frame_dir=paths['frame'],
#     json_dir=paths['keypoint'],
#     det_cfg  = str(BASE_DIR / "configs/detector/rtmdet_m_640-8xb32_coco-person_no_nms.py"),
#     det_ckpt = str(DATA_DIR / "checkpoints/sapiens/detector/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"),
#     pose_cfg = str(BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"),
#     pose_ckpt= str(DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"),
# )

# assign_sam_ids_to_keypoints(
#     sam_dir=paths['sam'],
#     kpt_dir=paths['keypoint'],
#     output_dir=paths['interp_data']
# )

generate_17kpt_skeleton_video(
    frame_dir=paths['frame'],
    kpt_dir=paths['interp_data'],
    output_path=str(f"{paths['interp_mp4']}.mp4")
)