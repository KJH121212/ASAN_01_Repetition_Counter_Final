import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path

# 1. 프로젝트 루트 경로를 시스템 경로에 추가 (utils 등을 임포트하기 위함)
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# 커스텀 유틸리티 임포트
from utils.path_list import path_list
from utils.extract_kpt import extract_id_keypoints, normalize_skeleton_array
from utils.counter_core import UniversalRepetitionCounter
from utils.generate_skeleton_video_v1 import generate_17kpt_skeleton_video
from utils.filter_id import filter_skeleton_by_ids

DATA_ROOT = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
csv_path = DATA_ROOT / "metadata_v2.1.csv"
bosan_path = DATA_ROOT / "bosanjin_seg_data.csv"
target = "biceps_curl"

df = pd.read_csv(csv_path)
bosan_df = pd.read_csv(bosan_path)
# 처리 결과를 저장할 리스트
results = []
total_rows = len(df)

print(f"🎬 총 {total_rows}개의 데이터 처리를 시작합니다.")

for idx in range(986,total_rows):
    # 1. end="\r"를 제거하여 tqdm 로그와 섞이지 않게 합니다.
    print(f"\n" + "="*50)
    print(f"🔄 [{idx + 1} / {total_rows}] 데이터 처리 시작")
    
    try:
        # 1. 현재 행의 정보 가져오기
        row = df.iloc[idx]
        patient_id = row['patient_id']
        common_path = row['common_path']

        # patient_id가 None(NaN)이거나 유효하지 않으면 건너뜁니다.
        if pd.isna(patient_id) or patient_id == "None":
            print(f"\n⚠️  [Skip] Index {idx}: patient_id가 유효하지 않습니다.")
            continue

        # 2. 경로 생성 (path_list 활용)
        paths = path_list(str(common_path))
        kpt_dir = str(paths['keypoint'])
        frame_dir = str(paths['frame'])
        out_json_dir = str(paths['interp_data'])
        output_video_path = f"{paths['interp_mp4']}.mp4"

        # 3. 특정 ID만 남기고 JSON 필터링
        filter_skeleton_by_ids(
            input_path=kpt_dir,
            output_path=out_json_dir,
            target_ids=[patient_id]
        )

        # 4. 필터링된 데이터를 바탕으로 스켈레톤 영상 생성
        generate_17kpt_skeleton_video(
            frame_dir=frame_dir,
            kpt_dir=out_json_dir,
            output_path=output_video_path
        )

        results.append({"index": idx, "patient_id": patient_id, "status": "Success"})

    except Exception as e:
        print(f"\n❌ [Error] Index {idx} 오류 발생: {e}")
        results.append({"index": idx, "patient_id": patient_id, "status": f"Fail: {str(e)}"})

# 최종 결과 출력
print(f"\n" + "="*40)
summary_df = pd.DataFrame(results)
success_count = len(summary_df[summary_df['status'] == 'Success'])
fail_count = len(summary_df[summary_df['status'] != 'Success'])
print(f"✅ 처리 완료: {success_count}건")
print(f"❌ 처리 실패: {fail_count}건")
print("="*40)