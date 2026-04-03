from pathlib import Path # 경로 처리를 위한 Path 객체 임포트
import pandas as pd # 데이터 처리를 위한 판다스 임포트

# 1. 경로 설정
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data") # 데이터 루트 경로
metadata_csv_path = DATA_DIR / "metadata_v2.1.csv" # 메타데이터 파일 경로

# 2. CSV 파일 로드 (patient_id의 3.0 현상을 방지하기 위해 문자열로 로드)
df_meta = pd.read_csv(metadata_csv_path, dtype={'patient_id': str}) # patient_id를 문자열 타입으로 지정

# 3. 수정 설정 (수정하고 싶은 모든 변수를 여기에 입력하세요)
target_row = 52

# 수정할 컬럼과 새로운 값 정의
updates = {
    "frames_done": True,     # 프레임 추출 완료 여부
    "sapiens_done": True,    # Sapiens 처리 완료 여부
    "reextract_done": True,  # 재추출 완료 여부
    "overlay_done": True,    # 오버레이 완료 여부
    "sam_done": True,        # SAM 처리 완료 여부
    "id_done": True,         # ID 할당 완료 여부
    "is_train": True,       # 학습 셋 포함 여부
    "is_val": False,         # 검증 셋 포함 여부
    "patient_id": "1"        # 환자 ID (문자열로 입력하여 3.0 방지)
}

# 4. 데이터 수정 및 비교 로직
if target_row < len(df_meta):
    # 식별용 경로 정보 추출
    v_path = df_meta.at[target_row, 'video_path']
    c_path = df_meta.at[target_row, 'common_path']
    
    # 수정 전 데이터 보관 (비교용)
    edit_cols = list(updates.keys())
    before_values = df_meta.loc[target_row, edit_cols].copy()

    # 데이터 실제 업데이트
    for col, new_value in updates.items():
        # 컬럼이 없으면 생성하고 값 입력
        df_meta.at[target_row, col] = new_value

    # 5. 결과 시각화 출력
    print(f"\n📍 대상 Video: {v_path}")
    print(f"📍 대상 Path: {c_path}")
    print(f"🔍 [Row {target_row}] 전체 컬럼 수정 결과")
    print("-" * 70)
    print(f"{'Column Name':<20} | {'Before (이전)':<20} | {'After (현재)':<20}")
    print("-" * 70)
    
    for col in edit_cols:
        b_val = str(before_values[col])
        a_val = str(df_meta.at[target_row, col])
        print(f"{col:<20} | {b_val:<20} | {a_val:<20}")
    print("-" * 70)

    # 6. 파일 업데이트 저장
    df_meta.to_csv(metadata_csv_path, index=False)
    print(f"✅ 모든 변수가 성공적으로 업데이트 및 저장되었습니다.")

else:
    print(f"⚠️ 에러: {target_row}번 행을 찾을 수 없습니다.")