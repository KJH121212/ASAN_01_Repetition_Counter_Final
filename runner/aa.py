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

# =================================================================
# 3. 데이터 정제 및 필터링 (is_train or is_val == True)
# =================================================================

# --- 3.1 Metadata 데이터프레임 정제 ---
# is_train 컬럼을 문자열로 바꾼 뒤, 공백을 제거하고 소문자로 만들어 'true'인지 확인합니다. (불리언 변환)
meta_df['is_train'] = meta_df['is_train'].astype(str).str.strip().str.lower() == 'true' # Train 값을 True/False로 확정합니다.
meta_df['is_val'] = meta_df['is_val'].astype(str).str.strip().str.lower() == 'true'     # Val 값을 True/False로 확정합니다.

# Train 혹은 Val 중 하나라도 True인 행만 선택하여 새로운 데이터프레임에 복사합니다.
df_meta_valid = meta_df[(meta_df['is_train'] == True) | (meta_df['is_val'] == True)].copy() # 유효한 메타데이터만 복사합니다.


# --- 3.2 Bosanjin Seg 데이터프레임 정제 ---
# 동일한 방식으로 Bosanjin 데이터의 공백 및 문자열 문제를 해결합니다.
bosan_df['is_train'] = bosan_df['is_train'].astype(str).str.strip().str.lower() == 'true' # Train 값을 True/False로 확정합니다.
bosan_df['is_val'] = bosan_df['is_val'].astype(str).str.strip().str.lower() == 'true'     # Val 값을 True/False로 확정합니다.

# Train 혹은 Val 중 하나라도 True인 행만 선택하여 새로운 데이터프레임에 복사합니다.
df_bosan_valid = bosan_df[(bosan_df['is_train'] == True) | (bosan_df['is_val'] == True)].copy() # 유효한 보산진 데이터만 복사합니다.


# --- 3.3 정제 결과 확인 ---
print(f"✅ 정제 완료: Metadata 유효 행 수 = {len(df_meta_valid)}") # 필터링된 메타데이터 개수를 출력합니다.
print(f"✅ 정제 완료: Bosanjin 유효 행 수 = {len(df_bosan_valid)}") # 필터링된 보산진 데이터 개수를 출력합니다.