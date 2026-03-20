import sys
import pandas as pd
from pathlib import Path

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
print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSANJIN_PATH.name})")
meta_df = pd.read_csv(METADATA_PATH)
bosan_df = pd.read_csv(BOSANJIN_PATH)

target = 15

# metadata  사용시
common_path = meta_df.iloc[target]['common_path']
start_frame = bosan_df.iloc[target]['start_frame']
end_frame = bosan_df.iloc[target]['end_frame']

# bosanjin_data 활용시
common_path = bosan_df.iloc[target]['common_path']
start_frame = bosan_df.iloc[target]['start_frame']
end_frame = bosan_df.iloc[target]['end_frame']

paths = path_list(common_path)
patient_id = int(meta_df.loc[meta_df['common_path'] == common_path, 'patient_id'].iloc[0]) # 조건에 맞는 첫 번째 ID 값을 안전하게 꺼내어 정수형(int)으로 변환합니다.

import shutil # 폴더와 그 내부의 모든 파일을 한 번에 삭제하기 위해 shutil 모듈을 임포트합니다.

# =================================================================
# 데이터 필터링 로직 (Bosanjin 제외 및 train/val 모두 False)
# =================================================================

# 1. 'common_path' 열의 문자열에 'Bosanjin'이 포함되지 않은(~) 행을 찾아 불리언 마스크를 생성합니다.
condition_not_bosanjin = ~meta_df['common_path'].str.contains('Bosanjin', na=False) 

# 2. 'is_train' 열과 'is_val' 열의 값이 모두(&) 명확하게 False인 행을 찾아 불리언 마스크를 생성합니다.
condition_both_false = (meta_df['is_train'] == False) & (meta_df['is_val'] == False) 

# 3. 위에서 정의한 두 가지 조건을 모두 만족(&)하는 행만 골라내어 최종 타겟 데이터프레임을 만듭니다.
target_df = meta_df[condition_not_bosanjin & condition_both_false] 

# 필터링이 잘 되었는지 확인하기 위해 결과 데이터프레임의 총 행 개수를 출력해 봅니다.
print(f"🔍 필터링 완료: 총 {len(target_df)}개의 데이터가 삭제 대상으로 선정되었습니다.")

deleted_dir_count = 0 # 정상적으로 삭제된 폴더의 개수를 카운트하기 위한 변수입니다.
deleted_file_count = 0 # 정상적으로 삭제된 MP4 파일의 개수를 카운트하기 위한 변수입니다.

# 2) 필터링된 대상 데이터프레임을 순회하며 파일과 폴더를 삭제합니다.
for index, row in target_df.iterrows(): # target_df의 각 행의 인덱스와 데이터를 반복해서 가져옵니다.
    current_common_path = row['common_path'] # 현재 행의 common_path 문자열을 변수에 할당합니다.
    
    try: # path_list 함수 호출 시 발생할 수 있는 예상치 못한 에러를 방지합니다.
        paths = path_list(current_common_path) # utils 모듈의 함수를 호출하여 필요한 경로 딕셔너리를 반환받습니다.
    except Exception as e: # 경로 생성 중 에러가 나면 잡아서 로그를 띄웁니다.
        print(f"⚠️ {current_common_path} 경로 생성 중 에러 발생: {e}") # 에러 내용을 출력하고 다음 루프로 넘어갑니다.
        continue # 아래 삭제 로직을 실행하지 않고 다음 행으로 안전하게 건너뜁니다.
        
    interp_data_dir = Path(paths.get('interp_data', '')) # 딕셔너리에서 폴더 경로를 가져와 Path 객체로 래핑합니다.
    interp_mp4_file = Path(f"{paths.get('interp_mp4', '')}.mp4") # 딕셔너리에서 파일 경로를 가져오고 .mp4 확장자를 붙여 Path 객체로 래핑합니다.

    # 3-A) 보간 데이터 폴더 삭제
    if interp_data_dir.exists() and interp_data_dir.is_dir(): # 해당 경로가 실제로 존재하며 폴더인지 검증합니다.
        shutil.rmtree(interp_data_dir) # 폴더 안의 내용물이 있더라도 통째로 안전하게 삭제합니다.
        deleted_dir_count += 1 # 폴더 삭제 카운트를 1 증가시킵니다.
        print(f"🗑️ 폴더 삭제됨: {interp_data_dir}") # 어떤 폴더가 삭제되었는지 사용자에게 보여줍니다.

    # 3-B) MP4 비디오 파일 삭제
    if interp_mp4_file.exists() and interp_mp4_file.is_file(): # 해당 경로가 실제로 존재하며 파일인지 검증합니다.
        interp_mp4_file.unlink() # Path 객체의 unlink() 메서드를 사용하여 해당 파일을 삭제합니다.
        deleted_file_count += 1 # 파일 삭제 카운트를 1 증가시킵니다.
        print(f"🗑️ 파일 삭제됨: {interp_mp4_file}") # 어떤 파일이 삭제되었는지 사용자에게 보여줍니다.

# 모든 작업이 끝난 후 최종 결과를 요약해줍니다.
print(f"✅ 정리 완료: 총 폴더 {deleted_dir_count}개, 파일 {deleted_file_count}개가 시스템에서 삭제되었습니다.")