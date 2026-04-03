import os # 운영체제와의 상호작용을 위해 os 모듈을 불러옵니다.
import sys # 시스템 경로 조작을 위해 sys 모듈을 불러옵니다.
import pandas as pd # 데이터프레임 조작과 결측치 처리를 위해 판다스를 불러옵니다.
from pathlib import Path # 파일 및 디렉토리 경로를 객체로 다루기 위해 Path를 사용합니다.

# =================================================================
# 1. 설정 및 커스텀 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from yolo.step1_dataset_builder import create_yolo_dataset_structure # 앞서 작성해둔 데이터셋 생성 함수를 불러옵니다.

if __name__ == "__main__": # 이 스크립트 파일이 직접 실행될 때만 메인 로직이 작동하도록 보호합니다.
    
    # =================================================================
    # 2. 경로 설정
    # =================================================================
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data") # 원본 데이터가 위치한 루트 폴더입니다.
    METADATA_PATH = DATA_DIR / "metadata_v2.0.csv" # 일반 영상에 대한 메타데이터 CSV 파일 경로입니다.
    BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data_v2.0.csv" # 구간별로 쪼개진 Bosanjin CSV 파일 경로입니다.
    TEST_DATASET_DIR = DATA_DIR / "6_YOLO_TRAINING_DATA/v2.0_step30" # YOLO 포맷으로 데이터가 복사/링크될 타겟 폴더입니다.
    SAMPLING_STEP = 30 # 추출할 프레임 간격을 설정합니다.

    # =================================================================
    # 3. 데이터 로드 및 전처리 
    # =================================================================
    print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSANJIN_PATH.name})") # 로드 작업 시작을 알립니다.
    meta_df = pd.read_csv(METADATA_PATH) # 일반 메타데이터를 데이터프레임으로 읽어옵니다.
    bosan_df = pd.read_csv(BOSANJIN_PATH) # Bosanjin 구간 데이터를 데이터프레임으로 읽어옵니다.

    columns_needed = ['common_path', 'start_frame', 'end_frame', 'is_train', 'is_val'] # 추출할 타겟 컬럼 목록입니다.

    meta_df['start_frame'] = pd.NA # 일반 데이터에는 시작 프레임이 없으므로 결측치로 초기화합니다.
    meta_df['end_frame'] = pd.NA # 일반 데이터에는 종료 프레임이 없으므로 결측치로 초기화합니다.
    
    meta_subset = meta_df[columns_needed].copy() # 일반 영상의 서브셋을 복사하여 만듭니다.
    bosan_subset = bosan_df[columns_needed].copy() # Bosanjin 구간 영상의 서브셋을 복사하여 만듭니다.

    condition_not_bosan = ~meta_subset['common_path'].str.contains('Bosanjin', na=False) # Bosanjin 텍스트가 없는 행만 마스킹합니다.
    meta_filtered = meta_subset[condition_not_bosan] # 필터링된 순수 일반 영상 데이터프레임만 남깁니다.

    combined_df = pd.concat([meta_filtered, bosan_subset], ignore_index=True) # 일반 영상과 Bosanjin 영상을 하나로 병합합니다.

    # 🌟 [핵심 개선: 띄어쓰기 및 자료형 오염 방어 로직] 🌟
    # CSV에 'True ' 처럼 공백이 있거나 문자열로 저장되었더라도 완벽하게 파싱하도록 강제 정규화합니다.
    combined_df['is_train'] = combined_df['is_train'].astype(str).str.strip().str.lower() == 'true' # 텍스트 변환 -> 공백 제거 -> 소문자 변환 후 'true'와 같은지 비교하여 순수 Boolean으로 만듭니다.
    combined_df['is_val'] = combined_df['is_val'].astype(str).str.strip().str.lower() == 'true' # is_val 열에 대해서도 똑같이 순수 Boolean 변환을 적용합니다.

    # 이제 자료형이 완벽해졌으므로 안전하게 필터링을 수행합니다.
    target_df = combined_df[combined_df['is_train'] | combined_df['is_val']].copy() # 학습용 또는 검증용 데이터만 최종 추출합니다.

    # 이제 증발하는 데이터 없이 정확하게 830개가 출력될 것입니다!
    print(f"🎯 처리 대상 시퀀스: 총 {len(target_df)}개 (Train + Val)") # 계산된 타겟 데이터 개수를 출력합니다.

    # =================================================================
    # 4. 데이터셋 생성 함수 실행
    # =================================================================
    generated_yaml = create_yolo_dataset_structure( # 데이터셋 빌드 함수를 호출합니다.
        df=target_df, # 조립된 최종 데이터프레임을 전달합니다.
        dataset_dir=TEST_DATASET_DIR, # 결과물을 저장할 타겟 경로를 전달합니다.
        data_dir=DATA_DIR, # 원본 데이터가 있는 경로를 전달합니다.
        step=SAMPLING_STEP # 프레임 샘플링 간격을 전달합니다.
    )
    
    print(f"\n✅ 모든 작업이 끝났습니다. 학습을 시작할 준비가 되었습니다! (설정 파일: {generated_yaml})") # 완료 메시지를 출력합니다.