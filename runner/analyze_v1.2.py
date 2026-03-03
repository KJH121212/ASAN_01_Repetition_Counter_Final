import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path

# 1. 프로젝트 루트 경로를 시스템 경로에 추가 (utils 등을 임포트하기 위함)
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# 커스텀 유틸리티 임포트
from utils.path_list import path_list
from utils.extract_kpt import extract_id_keypoints, normalize_skeleton_array
from utils.counter_core import UniversalRepetitionCounter
from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video
from utils.filter_id import filter_skeleton_by_ids

DATA_ROOT = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
csv_path = DATA_ROOT / "metadata_v2.1.csv"
bosan_path = DATA_ROOT / "bosanjin_seg_data.csv"

df = pd.read_csv(csv_path)
bosan_df = pd.read_csv(bosan_path)

target = 117
paths = path_list(str(df.iloc[target]['common_path']))
kpt_np = extract_id_keypoints(
    json_dir=paths["keypoint"],
    target_id=df.iloc[target]['patient_id']
)

norm_np = normalize_skeleton_array(kpt_np)

print(df.iloc[target]['common_path'])

from utils.postprocessing import apply_custom_kalman, apply_interpolation_outlier_filter, apply_sam_mask_outlier_filter

# ======================================================================
# 🚀 2. 실제 적용 및 그래프 그리기
# ======================================================================

# 🌟 1) 원본 픽셀 좌표 필터링: 스케일이 크므로 threshold와 노이즈 값을 크게 줍니다.
kpt_origin_filtered = apply_custom_kalman(
    kpt_np, 
    threshold=50.0, # 픽셀 단위: 예측보다 30픽셀 이상 튀면 Outlier 처리
    q_std=5.0,      # 움직임의 유연성
    r_std=1.0       # 측정값의 신뢰도
)

# 🌟 2) 정규화 좌표 필터링: 값이 작으므로 threshold와 노이즈 값을 작게 줍니다.
kpt_norm_filtered = apply_custom_kalman(
    norm_np,
    threshold=1,  # 정규화 단위: 예측보다 0.1 이상 튀면 Outlier 처리
    q_std=0.01,
    r_std=0.05
)

kpt_filter_normed = normalize_skeleton_array(kpt_origin_filtered)

kpt_filter_double = apply_custom_kalman(
    kpt_filter_normed,
    threshold=1,  # 정규화 단위: 예측보다 0.1 이상 튀면 Outlier 처리
    q_std=0.01,
    r_std=0.05
)

iqr_kpt = apply_interpolation_outlier_filter(
    data_np=kpt_origin_filtered,
    max_pixel_speed=150.0, # 1프레임당 최대 150픽셀 이동 허용
    use_iqr=True,
    iqr_multiplier=5.0 # 평소 속도 대비 5배 이상
)

sam_kpt = apply_sam_mask_outlier_filter(
    data_np=kpt_origin_filtered,
    sam_dir=paths['sam'],
    patient_id=df.iloc[target]['patient_id']
)

# --- 시각화 모듈 불러오기 ---
from utils.kpt_analysis_plot import plot_and_save_12kpt_analysis
from utils.generate_skeleton_video_v1 import generate_12kpt_skeleton_video_from_np

print("📊 원본 데이터 시각화 저장 중...")
plot_and_save_12kpt_analysis(
    data_array=kpt_np,
    save_path="./img/origin.png",
)

# print("📊 정규화 데이터 시각화 저장 중...")
# plot_and_save_12kpt_analysis(
#     data_array=norm_np,
#     save_path="./img/normalized.png",
# )

print("📊 원본 필터링 데이터 시각화 저장 중...")
plot_and_save_12kpt_analysis(
    data_array=kpt_origin_filtered,
    save_path="./img/origin_filtered.png",
)

print("📊 IQR 필터링 데이터 시각화 저장 중...")
plot_and_save_12kpt_analysis(
    data_array=iqr_kpt,
    save_path="./img/iqr_filtered.png",
)

print("📊 SAM 마스크 필터링 데이터 시각화 저장 중...")
plot_and_save_12kpt_analysis(
    data_array=sam_kpt,
    save_path="./img/sam_filtered.png"
)

# print("📊 정규화 필터링 데이터 시각화 저장 중...")
# plot_and_save_12kpt_analysis(
#     data_array=kpt_norm_filtered,
#     save_path="./img/normalized_filtered.png"
# )

# print("📊 정규화 필터링 데이터 시각화 저장 중...")
# plot_and_save_12kpt_analysis(
#     data_array=kpt_filter_normed,
#     save_path="./img/origin_filtered_normalized.png"
# )

# print("📊 정규화 필터링 데이터 시각화 저장 중...")
# plot_and_save_12kpt_analysis(
#     data_array=kpt_filter_double,
#     save_path="./img/origin_filtered_double.png"
# )

generate_12kpt_skeleton_video_from_np(
    frame_dir=paths["frame"],
    kpt_np=sam_kpt,
    output_path="./img/sam.mp4"
)




print("✅ 모든 처리가 완료되었습니다!")