import sys
import pandas as pd
from pathlib import Path

# 1. 경로 설정
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
CSV_PATH = DATA_DIR / "metadata_v1.0.csv"

# 2. 모듈 불러오기 경로 추가
sys.path.append(str(BASE_DIR))
try:
    from ground_truth_pipeline.step1_extract_frames import extract_frames
except ImportError:
    print("❌ 'ground_truth_pipeline' 모듈을 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit()

# 3. 데이터 읽기
if not CSV_PATH.exists():
    print(f"❌ 메타데이터 파일이 없습니다: {CSV_PATH}")
    sys.exit()

df = pd.read_csv(CSV_PATH)

# 컬럼 초기화
if 'frames_done' not in df.columns:
    df['frames_done'] = False
if 'n_frames' not in df.columns:
    df['n_frames'] = 0

total_count = len(df)
print(f"총 {total_count}개의 데이터를 확인합니다.")

# 4. 반복문 실행
count = 0

for idx in df.index:
    # 이미 작업이 완료된 행은 건너뜀
    is_done = str(df.loc[idx, "frames_done"]).lower() == "true"
    
    if is_done:
        continue

    # 데이터 추출
    common_name = df.loc[idx, "common_path"]
    video_path = Path(df.loc[idx, "video_path"])
    frame_dir = DATA_DIR / "1_FRAME" / common_name

    # [수정됨] tqdm 대신 현재 진행 상황 출력
    print(f"[{idx+1}/{total_count}] Processing: {common_name}")

    try:
        # 프레임 추출 함수 실행
        n_frames = extract_frames(
            video_path=video_path,
            frame_dir=frame_dir
        )
        
        # 5. 결과 업데이트
        df.loc[idx, "frames_done"] = True
        df.loc[idx, "n_frames"] = n_frames
        
        count += 1
        
        # (선택) 10개마다 중간 저장 (너무 자주 저장하면 느려질 수 있음)
        df.to_csv(CSV_PATH, index=False)

    except Exception as e:
        print(f"   └── ❌ [Error] 처리 중 오류 발생: {e}")
        # 오류 발생 시 해당 부분은 False로 유지

# 6. 최종 저장
df.to_csv(CSV_PATH, index=False)
print("\n모든 작업이 완료되고 CSV가 저장되었습니다.")