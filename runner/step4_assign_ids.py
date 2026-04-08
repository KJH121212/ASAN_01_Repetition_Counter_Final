import sys
import pandas as pd
from pathlib import Path
# tqdm 제거

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
# 프로젝트 루트 경로 (사용자 환경에 맞게 수정)
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from ground_truth_pipeline.step4_assign_ids import assign_sam_ids_to_keypoints
from utils.path_list_d03 import path_list_d03
from utils.generate_skeleton_video_v2 import generate_integrated_video


DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
METADATA_PATH = DATA_DIR / "metadata_v1.0.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================
if not METADATA_PATH.exists():
    print(f"❌ 메타데이터 없음: {METADATA_PATH}")
    sys.exit(1)

df = pd.read_csv(METADATA_PATH)

# 'id_done' 컬럼 확인/생성
if 'id_done' not in df.columns:
    df['id_done'] = False
else:
    # 문자열 "True"/"False" 처리
    df['id_done'] = df['id_done'].apply(lambda x: str(x).lower() == 'true')

# 🎯 처리해야 할 대상 필터링 (id_done이 False인 것만)
target_indices = df[~df['id_done']].index
total_targets = len(target_indices)

print(f"🚀 총 {len(df)}개 중 {total_targets}개의 시퀀스를 처리합니다.")

# =================================================================
# 3. 반복 실행 로직 (Main Loop)
# =================================================================
# enumerate를 사용하여 진행 순서(step)를 표시
for step, idx in enumerate(target_indices):
    row = df.loc[idx]
    common_path = row['common_path']
    
    # 🟢 경로 생성
    paths = path_list_d03(str(common_path))
    
    # 진행 상황 로그 출력 (flush=True로 즉시 출력 보장)
    print(f"[{step+1}/{total_targets}] Index {idx}: {common_path} ... ", end="", flush=True)

    try:
        # 함수 호출
        processed_count = assign_sam_ids_to_keypoints(
            sam_dir=paths['sam'],
            kpt_dir=paths['keypoint'],
            output_dir=None      # None이면 원본 덮어쓰기
        )
        
        generate_integrated_video(
            frame_dir=paths['frame'],       
            output_path=paths['mp4'],
            skeleton_dir=paths['keypoint'],
            sam_dir=paths['sam'],
            conf_threshold=0
        )

        # 결과 처리
        if processed_count > 0:
            # 성공 표시
            df.at[idx, 'id_done'] = True
            
            # 🔥 즉시 저장
            df.to_csv(METADATA_PATH, index=False, encoding='utf-8-sig')
            
            print(f"✅ 완료 ({processed_count} files)")
            
        else:
            print("⚠️ 업데이트 없음")

    except Exception as e:
        print(f"\n❌ [{idx}] 에러 발생: {e}")
        continue

print("\n🎉 모든 작업이 종료되었습니다.")