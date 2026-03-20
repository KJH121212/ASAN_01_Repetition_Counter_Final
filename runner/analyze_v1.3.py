# metadata 와 bosanjis_seg_data 내부 파일 개수 비교 코드

import sys # 시스템 경로 및 파이썬 인터프리터 제어를 위해 sys 모듈을 불러옵니다.
import pandas as pd # 데이터 분석과 표(데이터프레임) 조작을 위해 pandas 라이브러리를 불러옵니다.
from pathlib import Path # 경로를 문자열이 아닌 안전한 객체로 다루기 위해 Path를 사용합니다.

# =================================================================
# 0. 헬퍼 함수: 완벽한 칸 맞춤 및 합계(Total) 출력
# =================================================================
def print_table_with_total_underline(df, title, has_usage_pct=False): # Pandas 정렬 로직을 활용해 표와 합계를 출력하는 함수입니다.
    df_copy = df.copy() # 원본 데이터가 훼손되지 않도록 복사본을 생성하여 작업합니다.
    
    total_series = df_copy.sum(numeric_only=True) # 숫자형 컬럼들의 합계를 미리 계산하여 Series로 저장합니다.
    
    if has_usage_pct and 'Original_Total' in df_copy.columns: # 활용률(%) 컬럼이 필요한 표인지 확인합니다.
        tot_val = total_series['Valid_Total'] # 합산된 전체 유효 데이터 개수를 가져옵니다.
        tot_ori = total_series['Original_Total'] # 합산된 전체 원본 데이터 개수를 가져옵니다.
        total_series['Usage_%'] = round((tot_val / tot_ori) * 100, 1) if tot_ori > 0 else 0.0 # 0 나누기 에러를 방지하며 전체 활용률을 재계산합니다.
    
    for col in df_copy.columns: # 데이터프레임의 모든 컬럼을 하나씩 순회합니다.
        if col != 'Usage_%': # 소수점이 필요한 활용률 컬럼이 아니라면
            df_copy[col] = df_copy[col].astype(int) # 본문 데이터를 깔끔한 정수형(int)으로 변환합니다.
            total_series[col] = int(total_series[col]) # 합계 데이터 역시 정수형으로 변환하여 소수점(.0)을 없앱니다.

    # Total 행을 데이터프레임 하단에 임시로 붙여서 Pandas가 전체 열 너비를 스스로 맞추도록 유도합니다.
    df_with_total = pd.concat([df_copy, pd.DataFrame([total_series], columns=df_copy.columns, index=['Total'])]) # 본문과 합계를 결합합니다.
    
    output_str = df_with_total.to_string() # 결합된 전체 표를 Pandas의 자동 정렬이 적용된 하나의 문자열로 뽑아냅니다.
    lines = output_str.split('\n') # 뽑아낸 문자열을 줄 바꿈(\n) 기준으로 쪼개어 리스트로 만듭니다.
    
    header = lines[0] # 리스트의 첫 번째 줄은 컬럼명이 있는 헤더입니다.
    data_rows = lines[1:-1] # 두 번째 줄부터 마지막의 직전 줄까지는 본문 데이터입니다.
    total_row = lines[-1] # 리스트의 맨 마지막 줄은 우리가 추가한 Total 합계 행입니다.
    
    line_width = max(len(l) for l in lines) # 리스트 내에서 가장 긴 줄의 글자 수를 계산하여 표의 전체 너비를 구합니다.
    sep = "-" * line_width # 계산된 너비만큼 하이픈(-)을 반복하여 꽉 차는 구분선을 만듭니다.

    print(f"\n[{title}]") # 전달받은 표 제목을 위아래 공백과 함께 출력합니다.
    print(sep) # 헤더 위를 덮는 상단 구분선을 그립니다.
    print(header) # 헤더(컬럼명)를 출력합니다.
    print(sep) # 헤더와 본문 사이의 구분선을 그립니다.
    for row in data_rows: # 본문 데이터 리스트를 순회합니다.
        print(row) # 본문 데이터를 한 줄씩 차례대로 출력합니다.
    print(sep) # 본문이 끝났음을 알리는 명확한 밑줄(구분선)을 그립니다.
    print(total_row) # 구분선 바로 밑에 Total 합계 행을 출력합니다.
    print(sep) # 표를 마무리하는 최하단 구분선을 그립니다.

# =================================================================
# 1. 헬퍼 함수: 데이터 정제 (불리언 오염 방어)
# =================================================================
def clean_boolean(df): # 텍스트나 공백으로 오염된 True/False 값을 순수 불리언으로 고치는 함수입니다.
    for col in ['is_train', 'is_val']: # 훈련용과 검증용 플래그 컬럼을 순회합니다.
        df[col] = df[col].astype(str).str.strip().str.lower() == 'true' # 문자열 변환 -> 양옆 공백 제거 -> 소문자 변환 후 'true'인지 엄격히 검사합니다.
    df['is_valid'] = df['is_train'] | df['is_val'] # 둘 중 하나라도 True면 활용 가능한 유효 데이터로 마킹합니다.
    return df # 정제가 완료된 데이터프레임을 반환합니다.

# =================================================================
# 2. 메인 실행부 (데이터 로드 및 통합)
# =================================================================
if __name__ == "__main__": # 스크립트가 직접 실행될 때만 아래 로직을 수행합니다.
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data") # 데이터가 저장된 베이스 경로를 지정합니다.
    METADATA_PATH = DATA_DIR / "metadata_v2.0.csv" # 일반 메타데이터 CSV 파일의 경로입니다.
    BOSAN_SEG_PATH = DATA_DIR / "bosanjin_seg_data_v2.0.csv" # Bosanjin 세그먼트 데이터 CSV 파일의 경로입니다.

    print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSAN_SEG_PATH.name})") # 로드 상태를 콘솔에 출력합니다.
    meta_df = pd.read_csv(METADATA_PATH) # 메타데이터 원본을 읽어옵니다.
    bosan_df = pd.read_csv(BOSAN_SEG_PATH) # Bosanjin 세그먼트 데이터를 읽어옵니다.

    meta_df = clean_boolean(meta_df) # 일반 데이터의 불리언 오염을 치료합니다.
    bosan_df = clean_boolean(bosan_df) # Bosanjin 데이터의 불리언 오염을 치료합니다.

    # [🌟 누락되었던 핵심 복구!: 카테고리/환자/동작/각도 파싱]
    meta_df['category'] = meta_df['common_path'].str.split('/').str[0] # 슬래시 기준으로 경로 첫 부분을 카테고리로 뺍니다.
    meta_df['patient'] = meta_df['common_path'].str.split('/').str[1] # 슬래시 기준으로 경로 두 번째 부분을 환자 ID로 뺍니다. (복구 완료!)
    meta_df['angle'] = meta_df['common_path'].str.split('/').str[-1].str.split('__').str[0] # 파일명에서 언더바 앞부분을 각도로 뺍니다.
    meta_df['action'] = meta_df['common_path'].str.split('/').str[-1].str.split('__').str[1] # 파일명에서 언더바 뒷부분을 동작명으로 뺍니다.

    bosan_df['category'] = bosan_df['common_path'].str.split('/').str[0] # Bosanjin 데이터도 카테고리를 파싱합니다.
    bosan_df['patient'] = bosan_df['common_path'].str.split('/').str[1] # Bosanjin 데이터도 환자 ID를 파싱합니다. (복구 완료!)
    bosan_df['angle'] = bosan_df['raw_label'].str.split('__').str[0] # raw_label에서 각도를 파싱합니다.
    bosan_df['action'] = bosan_df['raw_label'].str.split('__').str[1] # raw_label에서 동작명을 파싱합니다.

    # [데이터 통합]
    meta_subset = meta_df[meta_df['category'] != 'Won_Kim_research_at_Bosanjin'].copy() # 중복 카운팅을 막기 위해 일반 데이터에서 보산진 폴더를 제거합니다.
    df_full = pd.concat([meta_subset, bosan_df], ignore_index=True) # 필터링된 일반 데이터와 세분화된 보산진 데이터를 하나로 합칩니다.

    df_full['action'] = df_full['action'].fillna('Unknown') # 동작명이 비어있는 행은 에러를 막기 위해 'Unknown'으로 채웁니다.
    df_full['angle'] = df_full['angle'].fillna('Unknown') # 각도명이 비어있는 행 역시 'Unknown'으로 채워줍니다.

    target_df = df_full[df_full['is_valid']].copy() # 실제 학습 및 통계에 쓰일 유효 데이터만 솎아내어 타겟 데이터프레임으로 확정합니다.
    print(f"🎯 처리 대상 시퀀스: 총 {len(target_df)}개 (Train + Val)") # 데이터 유실 없이 정상적으로 필터링된 데이터 개수를 확인합니다.

    # =================================================================
    # 3. 활용 현황 분석 및 표 출력
    # =================================================================
    # [Table 1: 최상위 폴더별 데이터 활용 현황]
    table1 = df_full.groupby('category').agg( # 카테고리 컬럼을 기준으로 그룹화하여 집계합니다.
        Train_Only=('is_train', lambda x: (x & ~df_full.loc[x.index, 'is_val']).sum()), # 순수하게 학습에만 쓰이는 개수를 계산합니다.
        Val_Only=('is_val', lambda x: (x & ~df_full.loc[x.index, 'is_train']).sum()),   # 순수하게 검증에만 쓰이는 개수를 계산합니다.
        Valid_Total=('is_valid', 'sum'),                                              # 활용 가능한 전체 유효 개수입니다.
        Original_Total=('is_valid', 'count')                                          # 유효성과 무관하게 존재하는 물리적 전체 파일 수입니다.
    ).astype(int) # 계산된 결과들을 정수형으로 묶어줍니다.
    table1['Usage_%'] = (table1['Valid_Total'] / table1['Original_Total'] * 100).round(1) # 유효 데이터를 원본 데이터로 나누어 활용률을 계산합니다.
    print_table_with_total_underline(table1, "Table 1: 최상위 폴더별 데이터 활용 현황 (오름차순)", has_usage_pct=True) # 포맷팅 함수를 통해 표 1을 출력합니다.

    # [Table 2 & 3: 주요 카테고리 내부 환자별 상세 현황]
    target_categories = ['Won_Kim_research_at_Bosanjin', 'AI_dataset'] # 환자별 상세 분석을 수행할 주요 카테고리 리스트입니다.
    for i, cat_name in enumerate(target_categories, start=2): # 리스트를 순회하며 표 번호를 2부터 자동으로 부여합니다.
        sub_df = df_full[df_full['category'] == cat_name].copy() # 현재 분석 중인 카테고리 데이터만 떼어냅니다.
        if sub_df.empty: continue # 해당 카테고리에 데이터가 없다면 에러 방지를 위해 건너뜁니다.
        
        patient_table = sub_df.groupby('patient').agg( # 환자(patient) ID를 기준으로 그룹화합니다.
            Train_Only=('is_train', lambda x: (x & ~sub_df.loc[x.index, 'is_val']).sum()), # 환자별 학습 전용 데이터 수입니다.
            Val_Only=('is_val', lambda x: (x & ~sub_df.loc[x.index, 'is_train']).sum()),   # 환자별 검증 전용 데이터 수입니다.
            Valid_Total=('is_valid', 'sum'),                                            # 환자별 유효 데이터 수입니다.
            Original_Total=('is_valid', 'count')                                        # 환자별 원본 데이터 수입니다.
        ).astype(int) # 정수형으로 묶어줍니다.
        patient_table['Usage_%'] = (patient_table['Valid_Total'] / patient_table['Original_Total'] * 100).round(1) # 환자별 데이터 활용률을 계산합니다.
        print_table_with_total_underline(patient_table, f"Table {i}: {cat_name} 내부 환자별 상세 활용 현황", has_usage_pct=True) # 표 2와 3을 포맷팅하여 출력합니다.

    # [Table 4: 동작(Action)별 상세 현황]
    action_stats = target_df.groupby('action').agg( # 유효 데이터(target_df)를 동작 이름별로 그룹화합니다.
        Train_Only=('is_train', lambda x: (x & ~target_df.loc[x.index, 'is_val']).sum()), # 동작별 순수 학습용 데이터 개수입니다.
        Val_Only=('is_val', lambda x: (x & ~target_df.loc[x.index, 'is_train']).sum()),   # 동작별 순수 검증용 데이터 개수입니다.
        Total_Rows=('is_valid', 'count')                                                # 해당 동작이 쓰인 전체 횟수(합계)입니다.
    ).astype(int) # 정수형으로 묶어줍니다.
    print_table_with_total_underline(action_stats, "Table 4: 동작(Action)별 상세 활용 현황", has_usage_pct=False) # 비율 정보 없이 동작별 표 4를 출력합니다.

    # [Table 5: 촬영 각도(Angle)별 상세 현황]
    angle_stats = target_df.groupby('angle').agg( # 유효 데이터를 촬영 각도별로 그룹화합니다.
        Train_Only=('is_train', lambda x: (x & ~target_df.loc[x.index, 'is_val']).sum()), # 각도별 순수 학습용 데이터 개수입니다.
        Val_Only=('is_val', lambda x: (x & ~target_df.loc[x.index, 'is_train']).sum()),   # 각도별 순수 검증용 데이터 개수입니다.
        Total_Rows=('is_valid', 'count')                                                # 해당 각도가 쓰인 전체 횟수(합계)입니다.
    ).astype(int) # 정수형으로 묶어줍니다.
    print_table_with_total_underline(angle_stats, "Table 5: 촬영 각도(Angle)별 상세 활용 현황", has_usage_pct=False) # 각도별 표 5를 출력합니다.

    # =================================================================
    # 4. 최종 분석 요약 출력
    # =================================================================
    print("\n" + "="*80) # 전체 요약을 알리는 상단 굵은 구분선입니다.
    print(f"💡 최종 분석 요약: 전체 {len(target_df)}개 유효 시퀀스 중") # 모델 학습에 투입될 전체 유효 시퀀스의 개수입니다.
    print(f"   순수 학습용(Train Only): {action_stats['Train_Only'].sum()}개") # 최종 확보된 학습 세트(Train)의 볼륨입니다.
    print(f"   순수 검증용(Val Only): {action_stats['Val_Only'].sum()}개") # 최종 확보된 평가 세트(Val)의 볼륨입니다.
    print("="*80) # 전체 스크립트 실행 종료를 알리는 하단 굵은 구분선입니다.