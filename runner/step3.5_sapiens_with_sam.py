
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/metadata_v2.0.csv")

sys.path.append(str(BASE_DIR))

# Step 2, 3 함수 Import
try:
    from ground_truth_pipeline.step3_sapiens_with_sam import run_sapiens_batch_inference
except ImportError as e:
    print(f"[CRITICAL ERROR] Import 실패: {e}")
    sys.exit(1)

# ============================================================
# Main 실행부
# ============================================================

if __name__ == "__main__":
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    
    df = pd.read_csv(DATA_DIR / "metadata_v2.0.csv")
    for target in range(173,174):
        COMMON_PATH = df['common_path'][target]

        FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
        SAM_DIR = DATA_DIR / "8_SAM" / COMMON_PATH
        OUTPUT_DIR = DATA_DIR / "2_KEYPOINTS" / COMMON_PATH
        MP4_OUTPUT_DIR = DATA_DIR / "3_MP4" / f"{COMMON_PATH}.mp4"
        
        # FRAME_DIR = DATA_DIR / "walking_data/FRAME/lateral__walking__1"
        # SAM_DIR = DATA_DIR / "walking_data/sam/lateral__walking__1"
        # OUTPUT_DIR = DATA_DIR / "walking_data/sapiens_133kpt" 
        # MP4_OUTPUT_DIR = DATA_DIR / "walking_data" / f"lateral__walking_133kpt__1.mp4"


        # COCO 133점 기반    
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.3b-210e_coco_wholebody-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth"

        # COCO 17점 기반 0.3b
        CONFIG = BASE_DIR / "configs/sapiens/sapiens_0.3b-210e_coco-1024x768.py"
        CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.3b_coco_best_coco_AP_796.pth"

        # COCO 17점 기반 0.6b
        # CONFIG = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sapiens_labeling/configs/sapiens/sapiens_0.6b-210e_coco-1024x768.py"
        # CKPT = DATA_DIR / "checkpoints/sapiens/pose/sapiens_0.6b_coco_best_coco_AP_812.pth"

        print(f"\noutput_dir: {OUTPUT_DIR}\n")
        # batch_size를 32~64 정도로 높여보세요 (VRAM 허용 시)
        count = run_sapiens_batch_inference(FRAME_DIR, SAM_DIR, OUTPUT_DIR, CONFIG, CKPT, batch_size=36)
        print(f"✅ 완료: {count}개 JSON 생성")