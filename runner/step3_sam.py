import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
import traceback
import torch
import multiprocessing as mp

# --- 1. 환경 설정 ---
# PyTorch 메모리 단편화 방지 설정 (OOM 방지용)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 프로젝트 경로 추가 (사용자 환경에 맞게 수정)
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))
from utils.huggingface_login import login_to_huggingface

# --- 2. 경로 설정 (전역 변수) ---
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03")
METADATA_PATH = DATA_DIR / "metadata_v1.0.csv"  # 요청하신 v2.0 버전 사용
CHECKPOINT_PATH = DATA_DIR / "checkpoints/SAM3/sam3.pt"
BPE_PATH = DATA_DIR / "checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
ENV_PATH = BASE_DIR / ".env"

# ==============================================================================
# [Worker] 개별 비디오 처리 워커 (별도 프로세스에서 실행)
# ==============================================================================
def process_video_worker(common_path, start_frame_idx=0):
    """
    단일 비디오를 처리하고 종료하는 함수입니다.
    이 함수가 종료되면 프로세스가 소멸하며 GPU 메모리가 100% 반환됩니다.
    """
    try:
        # 워커 내부에서 라이브러리를 임포트하여 상태를 초기화 (GPU 컨텍스트 충돌 방지)
        import torch
        import gc
        from ground_truth_pipeline.step3_track_objects import detect_objects, run_bidirectional_tracking
        from sam3 import build_sam3_image_model
        from sam3.model_builder import build_sam3_video_model

        # 필요시 HuggingFace 로그인
        login_to_huggingface(ENV_PATH)

        # 경로 설정
        curr_frame_path = DATA_DIR / "1_FRAME" / common_path
        curr_output_path = DATA_DIR / "8_SAM" / common_path
        prompt = "person"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"   ▶ [Worker] 시작: {common_path}")

        # ------------------------------------------------------
        # [Step 1] 이미지 모델 로드 -> 객체 검출 -> 모델 해제
        # ------------------------------------------------------
        image_model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH, bpe_path=BPE_PATH)
        image_model.to(device)
        
        detection_res = detect_objects(
            str(curr_frame_path), prompt, start_frame_idx, 
            model=image_model
        )
        
        # 이미지 모델 정리 (메모리 해제)
        del image_model
        gc.collect()
        torch.cuda.empty_cache()

        if not detection_res:
            print(f"   ❌ [Worker] 객체가 검출되지 않음")
            return False

        # ------------------------------------------------------
        # [Step 2] 비디오 모델 로드 -> 트래킹 -> 모델 해제
        # ------------------------------------------------------
        video_model = build_sam3_video_model(checkpoint_path=CHECKPOINT_PATH, apply_temporal_disambiguation=True, device=device)
        
        run_bidirectional_tracking(
            str(curr_frame_path), detection_res, str(curr_output_path), start_frame_idx,
            model=video_model
        )
        
        # 비디오 모델 정리 (메모리 해제)
        del video_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"   ✅ [Worker] 처리 완료")
        return True

    except Exception as e:
        print(f"   🔥 [Worker Error] 오류 발생: {e}")
        traceback.print_exc()
        return False

# ==============================================================================
# [Main] 메인 프로세스 (작업 관리)
# ==============================================================================
def main():
    # 즉각적인 로그 출력을 위해 버퍼링 해제
    sys.stdout.reconfigure(line_buffering=True)

    # 1. 메타데이터 로드
    if not METADATA_PATH.exists():
        print(f"[ERROR] 메타데이터 파일 없음: {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    total_len = len(df)

    # 컬럼이 없으면 생성
    if "sam_done" not in df.columns:
        df["sam_done"] = False
    
    # 2. 전처리: 완료 상태 확인 (문자열/불리언 처리)
    def check_status(val):
        return str(val).lower() == 'true'

    is_sam_done = df['sam_done'].apply(check_status)

    # 3. [핵심] 해야 할 작업만 필터링 (sam_done이 False인 것만)
    target_indices = df[~is_sam_done].index.tolist()
    count_todo = len(target_indices)

    print("="*60)
    print(f"🚀 SAM3 배치 처리 (프로세스 격리 모드)")
    print(f"📂 메타데이터: {METADATA_PATH}")
    print(f"[INFO] 전체 비디오: {total_len}개")
    print(f"[INFO] 처리 대상(TODO): {count_todo}개")
    print("="*60)

    if count_todo == 0:
        print("\n🎉 모든 작업이 이미 완료되었습니다! 종료합니다.")
        return

    # 4. 프로세스 생성 컨텍스트 설정
    # 'spawn' 방식을 사용하여 GPU 컨텍스트 충돌을 방지합니다.
    ctx = mp.get_context('spawn')

    # 5. 필터링된 인덱스 순회
    for i, idx in enumerate(target_indices):
        row = df.loc[idx]
        common_path = row["common_path"]
        
        print(f"\n[{i+1}/{count_todo}] 처리 중: {common_path}")

        curr_frame_path = DATA_DIR / "1_FRAME" / common_path
        if not curr_frame_path.exists():
            print(f"   ⚠️ [Skip] 프레임 폴더 없음: {curr_frame_path}")
            continue

        start_time = time.time()

        # ⭐ 핵심: 별도 프로세스 생성 및 시작
        # 메인 프로세스는 여기서 대기(join)하고, 워커는 GPU 사용 후 소멸합니다.
        p = ctx.Process(target=process_video_worker, args=(common_path, 0))
        p.start()
        p.join() # 프로세스가 끝날 때까지 대기
        
        # 종료 코드 및 파일 생성 여부로 성공 확인
        sam_output_dir = DATA_DIR / "8_SAM" / common_path
        json_files = list(sam_output_dir.glob("*.json")) if sam_output_dir.exists() else []

        # exitcode 0은 정상 종료
        if p.exitcode == 0 and len(json_files) > 0:
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"   ✅ 메인: 성공 확인 ({minutes}분 {seconds:.2f}초)")
            
            # DataFrame 업데이트
            df.at[idx, "sam_done"] = True
            
            # 즉시 저장 (진행 상황 보존)
            df.to_csv(METADATA_PATH, index=False)
            print(f"   └── [Saved] 메타데이터 업데이트 완료.")
        else:
            print(f"   ❌ 메인: 실패 또는 비정상 종료 (ExitCode: {p.exitcode})")
            # 실패 시 'sam_done'을 True로 바꾸지 않음 (나중에 재시도 가능하도록)

    print("\n🏁 모든 작업 종료.")

if __name__ == "__main__":
    main()