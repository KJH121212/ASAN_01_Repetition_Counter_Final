#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_video_scan_first.py

📌 최적화 기능:
1. CSV를 먼저 읽어서 '처리해야 할 비디오'만 리스트업 (Filter)
2. 할 일이 있을 때만 모델 로드 (불필요한 로딩 방지)
3. 진행률을 '남은 작업량' 기준으로 표시
"""

import sys
from pathlib import Path
import pandas as pd
from mmpose.apis import init_model as init_pose_estimator

# ============================================================
# 1️⃣ 기본 설정
# ============================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/metadata_v2.0.csv")

sys.path.append(str(BASE_DIR))

# Step 2, 3 함수 Import
try:
    from ground_truth_pipeline.step2_extract_poses import extract_keypoints
    from ground_truth_pipeline.step2_refine_poses import reextract_missing_keypoints
except ImportError as e:
    print(f"[CRITICAL ERROR] Import 실패: {e}")
    sys.exit(1)

# ============================================================
# 2️⃣ 처리 로직 함수
# ============================================================
def process_single_row(row_series, pose_estimator):
    """
    Pandas Series(행)를 받아서 처리하고 업데이트된 딕셔너리를 반환
    """
    row = row_series.to_dict()
    
    common_name    = row["common_path"]
    video_path     = Path(row["video_path"])
    frame_dir      = DATA_DIR / "1_FRAME" / common_name
    keypoint_dir   = DATA_DIR / "2_KEYPOINTS" / common_name
    
    frame_dir.mkdir(parents=True, exist_ok=True)
    keypoint_dir.mkdir(parents=True, exist_ok=True)

    def is_done(val):
        return str(val).lower() == "true"

    run_sapiens   = not is_done(row.get("sapiens_done", False))
    run_reextract = not is_done(row.get("reextract_done", False))

    n_frames = 0
    
    # [STEP 2] Sapiens Keypoint Extraction
    if run_sapiens:
        extract_keypoints(
            str(frame_dir), str(keypoint_dir),
            det_cfg  = str(BASE_DIR / "configs/detector/rtmdet_m_640-8xb32_coco-person_no_nms.py"),
            det_ckpt = str(DATA_DIR / "checkpoints/sapiens/detector/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"),
            pose_cfg = str(BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"),
            pose_ckpt= str(DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"),
            device="cuda:0"
        )
        row["sapiens_done"] = True

    # [STEP 3] Re-extract Missing Frames
    if run_reextract:
        if frame_dir.exists():
            n_frames = len(list(frame_dir.glob("*.jpg")))
        
        reextract_missing_keypoints(
            file_name = video_path.name,
            frame_dir = str(frame_dir),
            json_dir  = str(keypoint_dir),
            n_extracted_frames = n_frames,
            pose_estimator = pose_estimator  # 🔥 미리 로드된 모델 사용
        )
        row["reextract_done"] = True

    # 메타데이터 업데이트
    row["n_frames"] = len(list(frame_dir.glob("*.jpg")))
    row["n_json"]   = len(list(keypoint_dir.glob("*.json")))

    return row

# ============================================================
# 3️⃣ 메인 실행 (Scan First -> Run)
# ============================================================
if __name__ == "__main__":
    # 출력 버퍼링 끄기 (로그 즉시 출력)
    sys.stdout.reconfigure(line_buffering=True)

    if not CSV_PATH.exists():
        print(f"[ERROR] metadata.csv 없음: {CSV_PATH}")
        sys.exit(1)

    # 1. CSV 로드
    df = pd.read_csv(CSV_PATH)
    total_len = len(df)

    # 2. 전처리: 완료 여부 확인
    def check_status(val):
        return str(val).lower() == 'true'

    is_sapiens_done = df['sapiens_done'].apply(check_status)
    is_reextract_done = df['reextract_done'].apply(check_status)

    # 3. [핵심] 해야 할 작업만 필터링 (인덱스 리스트 추출)
    target_indices = df[ (~is_sapiens_done) | (~is_reextract_done) ].index.tolist()
    count_todo = len(target_indices)

    print(f"[INFO] 전체 비디오: {total_len}개")
    print(f"[INFO] 처리 대상(TODO): {count_todo}개")

    if count_todo == 0:
        print("\n🎉 모든 작업이 이미 완료되었습니다! 종료합니다.")
        sys.exit(0)

    # 4. 🔥 [누락되었던 부분 추가] 모델 로딩 (1회)
    print("\n[INIT] Sapiens 모델 로딩 중...", flush=True)
    try:
        global_pose_estimator = init_pose_estimator(
            str(BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"),
            str(DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"),
            device="cuda:0",
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] 모델 로딩 실패: {e}")
        sys.exit(1)
        
    print("[INIT] 로딩 완료. 작업 시작.", flush=True)

    # 5. 필터링된 인덱스만 순회
    print(f"\n🚀 작업 시작 (총 {count_todo}건)")
    
    for i, idx in enumerate(target_indices):
        # 1. 원본 데이터프레임에서 행(Row) 가져오기
        row = df.loc[idx]
        video_name = row['common_path']
        
        print(f"\n[{i+1}/{count_todo}] 처리 시작: {video_name}", flush=True)
        
        try:
            # 2. 🔥 처리 함수 호출 (모델 객체 전달)
            updated_row_dict = process_single_row(row, global_pose_estimator)

            # 3. 결과 업데이트 (원본 df에 반영)
            for k, v in updated_row_dict.items():
                df.at[idx, k] = v

            # 4. 즉시 저장 (데이터 유실 방지)
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"   └── [Saved] 저장 완료.", flush=True)

        except Exception as e:
            print(f"   └── ❌ [ERROR] 실행 중 오류 발생: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    print("\n[🏁 완료] 모든 대상 처리 종료.")


    