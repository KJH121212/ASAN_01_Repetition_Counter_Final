import sys
import pandas as pd
from pathlib import Path

# =================================================================
# 1. 설정 및 데이터 로드
# =================================================================
# 경로 설정
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSAN_SEG_PATH = DATA_DIR / "bosanjin_seg_data.csv"

# 데이터 로드
print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSAN_SEG_PATH.name})")
meta_df = pd.read_csv(METADATA_PATH) # 메타데이터 원본 로드
bosan_df = pd.read_csv(BOSAN_SEG_PATH) # 보산진 세그먼트 데이터 로드

# =================================================================
# 2. 데이터 기본 정제 및 통합 전처리
# =================================================================

# [Metadata 정제]
meta_df['is_train'] = meta_df['is_train'].astype(str).str.strip().str.lower() == 'true' # Train 값을 불리언형으로 변환합니다.
meta_df['is_val'] = meta_df['is_val'].astype(str).str.strip().str.lower() == 'true'     # Val 값을 불리언형으로 변환합니다.
meta_df['is_valid'] = meta_df['is_train'] | meta_df['is_val'] # Train/Val 중 하나라도 해당되면 활용 가능 데이터로 간주합니다.

meta_df['category'] = meta_df['common_path'].str.split('/').str[0] # 경로의 첫 번째 마디를 카테고리로 지정합니다.
meta_df['patient'] = meta_df['common_path'].str.split('/').str[1]  # 경로의 두 번째 마디를 환자 ID로 지정합니다.
# 경로 끝부분에서 각도와 동작 추출 (예: frontal__squat__1)
meta_df['angle'] = meta_df['common_path'].str.split('/').str[-1].str.split('__').str[0] # 파일명에서 각도를 추출합니다.
meta_df['action'] = meta_df['common_path'].str.split('/').str[-1].str.split('__').str[1] # 파일명에서 동작명을 추출합니다.

# [Bosanjin 정제]
bosan_df['is_train'] = bosan_df['is_train'].astype(str).str.strip().str.lower() == 'true' # Train 값을 불리언형으로 변환합니다.
bosan_df['is_val'] = bosan_df['is_val'].astype(str).str.strip().str.lower() == 'true'     # Val 값을 불리언형으로 변환합니다.
bosan_df['is_valid'] = bosan_df['is_train'] | bosan_df['is_val'] # 활용 가능 데이터 여부를 체크합니다.

bosan_df['category'] = bosan_df['common_path'].str.split('/').str[0] # 경로에서 카테고리를 추출합니다.
bosan_df['patient'] = bosan_df['common_path'].str.split('/').str[1]  # 경로에서 환자 ID를 추출합니다.
# raw_label에서 각도와 동작 추출
bosan_df['angle'] = bosan_df['raw_label'].str.split('__').str[0] # raw_label에서 각도를 추출합니다.
bosan_df['action'] = bosan_df['raw_label'].str.split('__').str[1] # raw_label에서 동작명을 추출합니다.

# [데이터 통합] - AI_dataset(Meta)와 Bosanjin(Seg)을 하나의 분석 데이터셋으로 합칩니다.
meta_subset = meta_df[meta_df['category'] != 'Won_Kim_research_at_Bosanjin'] # 중복 방지를 위해 메타데이터에서 보산진 폴더를 제외합니다.
df_full = pd.concat([meta_subset, bosan_df], ignore_index=True) # 분석용 통합 데이터프레임을 생성합니다.

# =================================================================
# 3. 데이터 중복(Overlap) 체크 및 출력
# =================================================================
# 동일한 데이터가 Train과 Val 양쪽에 모두 체크된 경우를 찾습니다.
duplicates = df_full[df_full['is_train'] & df_full['is_val']]

print("\n" + "="*85)
if not duplicates.empty:
    print(f"⚠️ [데이터 중복 확인] Train과 Val에 동시 포함된 데이터가 {len(duplicates)}건 발견되었습니다.")
    print("-" * 85)
    for path in duplicates['common_path']:
        print(f"📍 중복 경로: {path}") # 중복 발생한 경로를 리스트업합니다.
else:
    print("✅ 중복된 데이터가 없습니다. (학습/검증 세트가 완벽히 분리됨)")
print("="*85)

# =================================================================
# 4. 활용 현황 분석 (배타적 집계: Both_True 제외하고 순수 데이터만)
# =================================================================

# [Table 1: 최상위 폴더별 데이터 활용 현황]
table1 = df_full.groupby('category').agg(
    Train_Only=('is_train', lambda x: (x & ~df_full.loc[x.index, 'is_val']).sum()), # 순수 학습 데이터 수
    Val_Only=('is_val', lambda x: (x & ~df_full.loc[x.index, 'is_train']).sum()),   # 순수 검증 데이터 수
    Valid_Total=('is_valid', 'sum'), # 전체 유효 데이터 (중복 포함)
    Original_Total=('is_valid', 'count') # 폴더 내 물리적 전체 데이터 수
).astype(int)
table1['Usage_%'] = (table1['Valid_Total'] / table1['Original_Total'] * 100).round(1) # 활용률을 계산합니다.
table1 = table1.sort_index(ascending=True) # 오름차순 정렬

print("\n[Table 1: 최상위 폴더별 데이터 활용 현황 (오름차순)]")
print("-" * 85)
print(table1)

# [Table 2~3: 주요 카테고리 내부 환자별 상세 현황]
target_categories = ['Won_Kim_research_at_Bosanjin', 'AI_dataset'] # 상세 분석 대상
for i, cat_name in enumerate(target_categories, start=2):
    sub_df = df_full[df_full['category'] == cat_name] # 해당 카테고리 데이터만 추출합니다.
    patient_table = sub_df.groupby('patient').agg(
        Train_Only=('is_train', lambda x: (x & ~sub_df.loc[x.index, 'is_val']).sum()),
        Val_Only=('is_val', lambda x: (x & ~sub_df.loc[x.index, 'is_train']).sum()),
        Valid_Total=('is_valid', 'sum'),
        Original_Total=('is_valid', 'count')
    ).astype(int)
    patient_table['Usage_%'] = (patient_table['Valid_Total'] / patient_table['Original_Total'] * 100).round(1)
    patient_table = patient_table.sort_index(ascending=True) # 오름차순 정렬
    
    print(f"\n[Table {i}: {cat_name} 내부 환자별 상세 활용 현황]")
    print("-" * 85)
    print(patient_table)

# [유효 데이터 대상 각도/동작 분석] - 실제 분석에 투입될 데이터만 집계합니다.
df_valid = df_full[df_full['is_valid'] == True].copy()

# [Table 4: 동작(Action)별 상세 현황]
action_stats = df_valid.groupby('action').agg(
    Train_Only=('is_train', lambda x: (x & ~df_valid.loc[x.index, 'is_val']).sum()),
    Val_Only=('is_val', lambda x: (x & ~df_valid.loc[x.index, 'is_train']).sum()),
    Total_Rows=('action', 'count')
).astype(int).sort_index(ascending=True)

print("\n[Table 4: 동작(Action)별 상세 활용 현황 (오름차순)]")
print("-" * 75)
print(action_stats)

# [Table 5: 촬영 각도(Angle)별 상세 현황]
angle_stats = df_valid.groupby('angle').agg(
    Train_Only=('is_train', lambda x: (x & ~df_valid.loc[x.index, 'is_val']).sum()),
    Val_Only=('is_val', lambda x: (x & ~df_valid.loc[x.index, 'is_train']).sum()),
    Total_Rows=('angle', 'count')
).astype(int).sort_index(ascending=True)

print("\n[Table 5: 촬영 각도(Angle)별 상세 활용 현황 (오름차순)]")
print("-" * 75)
print(angle_stats)

# 최종 요약
print("\n" + "="*85)
print(f"💡 최종 분석 요약: 전체 {len(df_valid)}개 유효 데이터 중")
print(f"   순수 학습용(Train Only): {action_stats['Train_Only'].sum()}개")
print(f"   순수 검증용(Val Only): {action_stats['Val_Only'].sum()}개")
print("="*85)