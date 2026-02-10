import pandas as pd
from pathlib import Path

# 1. 파일 경로 설정
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
csv_path = DATA_DIR / "metadata_v2.0.csv"

# 2. 데이터 로드
if not csv_path.exists():
    print(f"❌ 메타데이터 파일이 없습니다: {csv_path}")
else:
    df = pd.read_csv(csv_path)

    # 3. 필터링: is_train 또는 is_val이 True인 행만 추출
    # (학습이나 검증 중 하나라도 쓰이는 데이터)
    filtered_df = df[(df['is_train'] == True) | (df['is_val'] == True)]

    print(f"📊 [전체 메타데이터] 행 개수: {len(df)}")
    print(f"🎯 [사용 데이터 (Train+Val)] 행 개수: {len(filtered_df)}")
    print("-" * 60)

    # 4. 분석할 카테고리 리스트
    categories = [
        "AI_dataset",
        "Won_Kim_research_at_Bosanjin",
        "Nintendo_Therapy",
        "sample_data"
    ]

    # 5. 카테고리별 개수 확인
    print(f"{'Category':<35} | {'Total':<6} | {'Train':<6} | {'Val':<6}")
    print("-" * 60)

    total_check = 0
    
    for cat in categories:
        # 1) 전체 사용 데이터(Train+Val) 중 해당 카테고리 개수
        total_count = filtered_df['common_path'].str.contains(cat).sum()
        
        # 2) 상세 구분: Train 개수
        train_count = df[(df['is_train'] == True) & (df['common_path'].str.contains(cat))].shape[0]
        
        # 3) 상세 구분: Val 개수
        val_count = df[(df['is_val'] == True) & (df['common_path'].str.contains(cat))].shape[0]

        print(f"{cat:<35} | {total_count:<6} | {train_count:<6} | {val_count:<6}")
        total_check += total_count

    print("-" * 60)
    print(f"{'Sum of Counts':<35} | {total_check:<6}")