# sapiens로 뽑은 kpt를 사용하여 counting

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =================================================================
# 1. 경로 설정 및 모듈 임포트
# =================================================================
print("📋 [Step 0] 초기 설정 및 모듈 로드")
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data.csv"

sys.path.append(str(BASE_DIR))

try:
    from utils.path_list import path_list
    from utils.extract_kpt import extract_id_keypoints, normalize_skeleton_array
    from utils.counter_core import UniversalRepetitionCounter
    from utils.parser import parse_common_path
    from utils.generate_skeleton_video_v1 import generate_counting_skeleton_video
    print("✅ 모듈 로드 완료.")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    sys.exit(1)

# =================================================================

print("📋 [Step 1] 사용 데이터 처리")

df = pd.read_csv(CSV_PATH)
bosanjin_df = pd.read_csv(BOSANJIN_PATH)
# common_path에 biceps_curl이 포함된 행만 필터링
filtered_df = df[df['common_path'].str.contains('biceps_curl', case=False, na=False)]

parsed_results = filtered_df['common_path'].apply(parse_common_path)
filtered_df['angle'] = parsed_results.apply(lambda x: x[0])  # camera_angle
filtered_df['exercise'] = parsed_results.apply(lambda x: x[1])  # exercise_name

print(filtered_df.head())
print("파싱 완료")

# =================================================================
print("\n🚀 [Step 3] 개별 데이터 추출 및 렌더링 시작...")

# =================================================================
# 💡 [핵심] 테스트할 시퀀스 개수를 설정합니다!
# 3개만 돌려보려면 3을 넣고, 전체 데이터를 다 돌리려면 None으로 변경하세요.
# =================================================================
TEST_LIMIT  = 3

# TEST_LIMIT이 지정되어 있으면 그 개수만큼만 자르고, None이면 필터링된 전체 데이터를 사용합니다.
df_to_process = filtered_df.head(TEST_LIMIT) if TEST_LIMIT is not None else filtered_df

print(f"👉 총 {len(df_to_process)}개의 시퀀스를 처리합니다. (전체 데이터: {len(filtered_df)}개)")

# 선택된 데이터프레임만 순회합니다.
# =================================================================
# (앞부분 코드 동일) ...
# =================================================================

# 선택된 데이터프레임만 순회합니다.
for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="전체 시퀀스 처리"):
    
    # 1. 사용할 기본 정보 추출
    common_path = row['common_path']
    camera_angle = row['angle']   
    exercise_name = row['exercise'] 
    
    if pd.isna(row['patient_id']):
        print(f"\n⚠️ [건너뜀] Index {idx}: patient_id가 비어있습니다. ({common_path})")
        continue
        
    patient_id = int(float(row['patient_id']))
    paths = path_list(common_path, create_dirs=True)

    # 2. 데이터 추출 및 정규화
    try:
        kpt_np = extract_id_keypoints(paths['keypoint'], patient_id)
        if len(kpt_np) == 0:
            print(f"\n⚠️ [경고] Index {idx}: 추출된 프레임이 0개입니다.")
            continue
            
        norm_kpt_np = normalize_skeleton_array(kpt_np)

        # 💡 3. [콘솔 확인용] 카운터 객체 1
        console_counter = UniversalRepetitionCounter(exercise_name=exercise_name, camera_angle=camera_angle)

        for frame_idx, current_kpt in enumerate(norm_kpt_np):
            metrics, events = console_counter.process_frame(current_kpt)
            if events['left'] or events['right']:
                print(f"\n   -> [{frame_idx} 프레임] 카운트 증가! | 왼팔: {console_counter.counts['left']}회, 오른팔: {console_counter.counts['right']}회")

        print(f"\n🎯 [Index {idx}] 최종 결과: 왼팔 {console_counter.counts['left']}회 / 오른팔 {console_counter.counts['right']}회")
        print(f"✅ {common_path} 데이터 처리 완료!")
        print("-" * 50)

        # 💡 4. [비디오 렌더링용] 카운터 객체 새로 생성 및 비디오 함수 호출
        video_counter = UniversalRepetitionCounter(exercise_name=exercise_name, camera_angle=camera_angle)
        
        generate_counting_skeleton_video(
            frame_dir=str(paths['frame']),
            kpt_dir=str(paths['keypoint']), # 주의: 매개변수 이름을 kpt_dir로 맞췄습니다.
            output_path=str(paths['test'] / f"counter_id{patient_id}.mp4"),
            counter=video_counter,          # 새로 만든 0회짜리 카운터 전달
            patient_id=patient_id           # 타겟 환자 ID 전달
        )

    except Exception as e:
        print(f"\n❌ [오류] Index {idx} 처리 중 에러 발생: {e}")

print("\n✨ 모든 시퀀스 처리가 완료되었습니다!")