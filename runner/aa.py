import sys
import pandas as pd
from pathlib import Path

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from utils.path_list import path_list
from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video, generate_12kpt_skeleton_video_segment
from utils.extract_kpt import normalize_skeleton_array, extract_id_keypoints, save_only_target_kpt_json
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


for target in range(511,526):

    common_path = bosan_df.iloc[target]['common_path']
    start_frame = bosan_df.iloc[target]['start_frame']
    end_frame = bosan_df.iloc[target]['end_frame']

    paths = path_list(common_path)
    patient_id = int(meta_df.loc[meta_df['common_path'] == common_path, 'patient_id'].iloc[0]) # 조건에 맞는 첫 번째 ID 값을 안전하게 꺼내어 정수형(int)으로 변환합니다.

    mp4_output_path = paths['interp_mp4']/f"{bosan_df.iloc[target]['raw_label']}.mp4"
    mp4_test_path = paths['test']/f"seg_filtered/{bosan_df.iloc[target]['raw_label']}.mp4"


    print(common_path)
    print("\n",bosan_df.iloc[target]['raw_label'],"\n")

    kpt = extract_id_keypoints(
        json_dir=paths['keypoint'],
        target_id=patient_id,
        start_frame=start_frame,
        end_frame=end_frame
    )

    from utils.postprocessing import apply_axis_selective_kalman, apply_axis_velocity_kalman, apply_axis_selective_iqr_filter, apply_kalman_smoothing

   # 데이터 유효성 검사 (수정된 부분)
    if kpt is not None and hasattr(kpt, 'shape') and len(kpt.shape) == 3:
        # 1. Selective Kalman Filter
        filtered_kpt = apply_axis_selective_kalman(
            data_np=kpt,
            threshold=50,
            q_std=0.5,
            r_std=0.3,
            target_kpts=[0,1,2,3,4,5,6,7,8,9,10,11],
            axis='both'
        )
        
        # 2. Kalman Smoothing
        filtered_kpt = apply_kalman_smoothing(
            data_np=filtered_kpt,
            q_std=0.5,
            r_std=0.3,
            target_kpts=[0,1,2,3,4,5,6,7,8,9,10,11],
            axis='both'
        )
    else:
        # 변수명 수정: start -> start_frame, end -> end_frame
        print(f"⚠️ [SKIP] ID:{patient_id} 구간({start_frame}~{end_frame}) 데이터가 유효하지 않음 (Shape: {getattr(kpt, 'shape', 'None')})")
        continue

    
    from utils.kpt_analysis_plot import plot_and_save_12kpt_analysis

    plot_and_save_12kpt_analysis(
        data_array=kpt,
        save_path=paths['test']/f"plot_filtered/{bosan_df.iloc[target]['raw_label']}_origin.png"
    )

    plot_and_save_12kpt_analysis(
        data_array=filtered_kpt,
        save_path=paths['test']/f"plot_filtered/{bosan_df.iloc[target]['raw_label']}_filtered.png"
    )

    generate_12kpt_skeleton_video_segment(
        frame_dir=paths['frame'],
        kpt_np=filtered_kpt,
        output_path=mp4_test_path,
        start_idx=start_frame,
        end_idx=end_frame
    )

    # save_only_target_kpt_json(
    #     src_dir=paths['keypoint'],
    #     output_dir=paths['interp_data'],
    #     kpt_array=filtered_kpt,
    #     target_id=patient_id,
    #     start_frame=start_frame
    # )

    # generate_17kpt_skeleton_video(
    #     frame_dir=paths['frame'],
    #     kpt_dir=paths['interp_data'],
    #     output_path=mp4_output_path,
    #     start_idx=start_frame,
    #     end_idx=end_frame,
    #     conf_threshold=0
    # )
