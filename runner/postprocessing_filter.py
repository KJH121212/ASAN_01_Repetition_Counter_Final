import sys
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Union, Literal # Union과 필요한 타입들을 임포트합니다.

target = 213
update = False
# update = True

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from utils.path_list import path_list

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data_v2.1.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================
print(f"📂 CSV 로드 중...")
meta_df = pd.read_csv(METADATA_PATH)
bosan_df = pd.read_csv(BOSANJIN_PATH)

# metadata  사용시
common_path = meta_df.iloc[target]['common_path']
start_frame = 0
end_frame = None


# bosanjin_data 활용시
# common_path = bosan_df.iloc[target]['common_path']
# start_frame = bosan_df.iloc[target]['start_frame']
# end_frame = bosan_df.iloc[target]['end_frame']

paths = path_list(common_path)
patient_id = int(meta_df.loc[meta_df['common_path'] == common_path, 'patient_id'].iloc[0]) # 조건에 맞는 첫 번째 ID 값을 안전하게 꺼내어 정수형(int)으로 변환합니다.

# common path 확인
print(common_path)
# =================================================================
# 3. patient_id의 skeleton data를 numpy 형태로 추출
# =================================================================
from utils.extract_kpt import extract_id_keypoints
kpt_np = extract_id_keypoints(
    json_dir=paths['keypoint'],
    target_id=patient_id
)

# =================================================================
# 4. postprocessing (칼만, IQR, smoothing)
# =================================================================
from utils.postprocessing import apply_axis_selective_kalman,apply_axis_selective_iqr_filter, apply_kalman_smoothing, fix_keypoints_to_stat, apply_segment_interpolation

kpt_filtered=kpt_np

# kpt_filtered = fix_keypoints_to_stat(
#     kpt_filtered,
#     target_kpts=[0,1,2,3,4,5,7,9,11],
#     axis='both',
#     # method='binned_mode',
#     # bin_size=5,
#     # min_score=0
#     method='specific_frame',
#     frame_idx=165
# ) 

# kpt_filtered = fix_keypoints_to_stat(
#     kpt_filtered,
#     target_kpts=[1,3,5],
#     axis='both',
#     # method='binned_mode',
#     # bin_size=5,
#     # min_score=0
#     method='specific_frame',
#     frame_idx=205
# ) 

# kpt_filtered = apply_segment_interpolation(
#     data_np=kpt_filtered,
#     start_frame=244,
#     end_frame=262,
#     target_kpts=[5],
#     axis='both'
# )

# kpt_filtered = apply_axis_selective_iqr_filter(
#     data_np=kpt_filtered,
#     target_kpts=[8],
#     max_pixel_speed=40,
#     iqr_multiplier=3,
#     use_iqr=True,
#     axis='y'
# )

kpt_filtered = apply_axis_selective_kalman(
    data_np=kpt_filtered,
    target_kpts=[0,1,2,3,4,5,6,7,8,9,10,11],
    threshold=100,
    axis='both'    
)

kpt_filtered =  apply_kalman_smoothing(
    data_np=kpt_filtered,
    q_std=0.01,
    r_std=0.1,
    target_kpts=[0,1,2,3,4,5,6,7,8,9,10,11],
    axis='both'
)


# =================================================================
# 5 & 6. 결과 시각화 및 최종 저장 분기 처리
# =================================================================

# 필요한 모든 시각화 및 저장 관련 함수들을 코드 블록 상단에서 한 번에 임포트합니다.
from utils.kpt_analysis_plot import plot_and_save_12kpt_analysis # 시각화 함수를 임포트합니다.
from utils.generate_skeleton_video_v2 import generate_skeleton_video_np, generate_integrated_video # 테스트용 스켈레톤 비디오 생성 함수를 임포트합니다.
from utils.extract_kpt import save_patient_only_12_to_17 # JSON 키포인트 저장 함수를 임포트합니다.

# False면 5번(시각화 테스트) 실행, True면 6번(최종 데이터 저장)을 실행하기 위한 플래그 변수입니다.

if not update: # a가 False인 경우 (즉, 아직 테스트 및 시각화 확인 단계인 경우) 실행됩니다.
    print("📊 원본 및 필터링 데이터 시각화 저장 중...") # 사용자에게 시각화 진행 상태를 알립니다.
    
    plot_and_save_12kpt_analysis(       # 원본 데이터를 이미지로 시각화하여 저장하는 함수를 호출합니다.
        data_array=kpt_np,              # 원본 키포인트 배열을 입력으로 전달합니다.
        save_path="./img/origin.png",   # 결과물을 저장할 경로를 지정합니다.
    )
    
    plot_and_save_12kpt_analysis(       # 필터링된 데이터를 이미지로 시각화하여 저장하는 함수를 호출합니다.
        data_array=kpt_filtered,        # 필터링 처리가 끝난 배열을 입력으로 전달합니다.
        save_path="./img/filtered.png", # 결과물을 저장할 경로를 지정합니다.
    )
    
    generate_skeleton_video_np(         # 필터링된 데이터로 테스트용 스켈레톤 비디오를 생성합니다.
        frame_dir=paths['frame'],       # 원본 프레임 이미지가 있는 디렉토리 경로입니다.
        output_path="./img/test.mp4",   # 완성된 테스트 비디오를 저장할 경로입니다.
        skeleton_np=kpt_filtered,       # 비디오에 오버레이할 필터링된 키포인트 배열입니다.
        start_idx=0,                    # 비디오 생성을 시작할 프레임 인덱스입니다.
        end_idx=None,                   # 비디오 생성을 종료할 인덱스입니다 (None이면 끝까지).
        conf_threshold=0,               # 키포인트를 그릴 최소 신뢰도 임계값입니다.
        fps=60                          # 생성될 비디오의 초당 프레임 수(FPS)를 설정합니다.
    )

else: # a가 True인 경우 (즉, 필터링 결과가 우수하여 최종 저장을 결정한 경우) 실행됩니다.
    print("💾 필터링 결과가 우수하여 최종 데이터를 저장합니다...") # 최종 저장 프로세스 시작을 알립니다.
    
    save_patient_only_12_to_17(             # 타겟 환자의 키포인트만 17포인트 포맷으로 변환하여 저장합니다.
        src_dir=paths['keypoint'],          # 원본 키포인트 JSON 파일이 있는 폴더입니다.
        output_dir=paths['interp_data'],    # 필터링된 새 JSON을 저장할 대상 폴더입니다.
        kpt_array=kpt_filtered,             # 덮어쓸 필터링된 키포인트 배열 데이터입니다.
        patient_id=patient_id,              # 타겟으로 삼을 특정 환자의 고유 ID입니다.
        start_frame=start_frame             # 처리를 시작할 기준 프레임 인덱스입니다.
    )
    
    generate_integrated_video( # SAM 마스크 등과 통합된 최종 결과 비디오를 생성합니다.
        frame_dir=paths['frame'], # 원본 프레임 이미지가 있는 디렉토리 경로입니다.
        output_path=f"{paths['interp_mp4']}.mp4", # 최종 비디오 파일의 저장 경로 및 이름입니다.
        skeleton_dir=paths['interp_data'], # 방금 저장한 필터링된 키포인트 디렉토리 경로입니다.
        sam_dir=paths['sam'], # SAM(Segment Anything Model) 데이터가 있는 디렉토리입니다.
        start_idx=start_frame, # 통합 비디오 생성을 시작할 프레임 인덱스입니다.
        end_idx=end_frame, # 통합 비디오 생성을 종료할 프레임 인덱스입니다.
        conf_threshold=0, # 키포인트를 그릴 최소 신뢰도 임계값입니다.
        fps=30 # 생성될 비디오의 초당 프레임 수(FPS)를 설정합니다.
    )


    # # =====================================================
    # # 원본 변경
    # # =====================================================

    # save_patient_only_12_to_17(           # 타겟 환자의 키포인트만 17포인트 포맷으로 변환하여 저장합니다.
    #     src_dir=paths['keypoint'],        # 원본 키포인트 JSON 파일이 있는 폴더입니다.
    #     output_dir=paths['keypoint'],     # 필터링된 새 JSON을 저장할 대상 폴더입니다.
    #     kpt_array=kpt_filtered,           # 덮어쓸 필터링된 키포인트 배열 데이터입니다.
    #     patient_id=patient_id,            # 타겟으로 삼을 특정 환자의 고유 ID입니다.
    #     start_frame=start_frame           # 처리를 시작할 기준 프레임 인덱스입니다.
    # )

    # generate_integrated_video( # SAM 마스크 등과 통합된 최종 결과 비디오를 생성합니다.
    #     frame_dir=paths['frame'], # 원본 프레임 이미지가 있는 디렉토리 경로입니다.
    #     output_path=f"{paths['mp4']}", # 최종 비디오 파일의 저장 경로 및 이름입니다.
    #     skeleton_dir=paths['keypoint'], # 방금 저장한 필터링된 키포인트 디렉토리 경로입니다.
    #     sam_dir=paths['sam'], # SAM(Segment Anything Model) 데이터가 있는 디렉토리입니다.
    #     start_idx=start_frame, # 통합 비디오 생성을 시작할 프레임 인덱스입니다.
    #     end_idx=end_frame, # 통합 비디오 생성을 종료할 프레임 인덱스입니다.
    #     conf_threshold=0, # 키포인트를 그릴 최소 신뢰도 임계값입니다.
    #     fps=30 # 생성될 비디오의 초당 프레임 수(FPS)를 설정합니다.
    # )