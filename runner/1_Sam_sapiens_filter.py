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
from ground_truth_pipeline.step3_sapiens_with_sam import find_missing_instances, run_sapiens_inference_from_list
from ground_truth_pipeline.step4_assign_ids import assign_sam_ids_to_keypoints, filter_duplicate_skeletons

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================
print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSANJIN_PATH.name})")
meta_df = pd.read_csv(METADATA_PATH)
total_num = len(meta_df)
for target in range(1009,total_num):

    common_path = meta_df.iloc[target]['common_path']

    paths = path_list(common_path)
    patient_id = int(meta_df.loc[meta_df['common_path'] == common_path, 'patient_id'].iloc[0]) # 조건에 맞는 첫 번째 ID 값을 안전하게 꺼내어 정수형(int)으로 변환합니다.

    CONFIG = BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"
    CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"

    # 3. 메인 실행부
    print("=" * 60)
    print(f"📦 [JOB {target+1:04d}/{total_num}] INITIALIZING...")
    print(f"👤 Patient ID : {patient_id}")
    print(f"📂 Path       : {common_path}") # 경로가 길면 잘라서 표시
    print("-" * 60)

    assign_sam_ids_to_keypoints(
        sam_dir=paths['sam'],
        kpt_dir=paths['keypoint'],
        output_dir=paths['keypoint']
    )

    missing_list = find_missing_instances(
        sam_dir=paths['sam'],
        kpt_dir=paths['keypoint'],
        target_id=patient_id
    )

    run_sapiens_inference_from_list(
        missing_list=missing_list,
        frame_dir=paths['frame'],
        sam_dir=paths['sam'],
        output_dir=str(paths['keypoint']),
        config_path=CONFIG,
        ckpt_path=CKPT,
        target_id=patient_id,
        batch_size=8
    )

    generate_17kpt_skeleton_video(
        frame_dir=paths['frame'],
        kpt_dir=paths['keypoint'],
        output_path=paths['mp4']
    )
    
    print(f"✅ Finished Job {target+1:04d}")