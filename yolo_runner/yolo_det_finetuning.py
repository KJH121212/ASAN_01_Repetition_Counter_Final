import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv

# 1. 설정 로드
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
CONFIG_PATH = BASE_DIR / "configs/yolo/det_v2.0_step30.yaml"

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# 2. WandB 초기화
load_dotenv(BASE_DIR / ".env")
wandb.login(key=os.getenv("WANDB_API_KEY"))

# 💡 Pose 코드와 동일하게 init 세션을 엽니다.
wandb.init(
    project=cfg['project_name'],
    name=cfg['run_name'],
    config=cfg,
    dir=cfg['output']['base_dir'],
    resume="allow"
)

# 💡 [핵심] Detection Metrics를 WandB에 쏴주는 콜백 함수
def on_train_epoch_end(trainer):
    if wandb.run:
        # Detection 지표(box_loss, cls_loss, mAP 등)를 로그에 남김
        wandb.log(trainer.metrics)
        wandb.log({"epoch": trainer.epoch + 1})

# 3. 모델 로드
model = YOLO(cfg['model']['base_path']) 

# 💡 [핵심] 콜백 등록 (이게 있어야 그래프가 그려집니다)
model.add_callback("on_train_epoch_end", on_train_epoch_end)

print(f"🔥 Object Detection 학습 시작: {cfg['run_name']}")

# 4. 학습 시작
model.train(
    data=cfg['data']['config_path'],
    project=cfg['output']['base_dir'],
    name=cfg['run_name'],
    cache=False,
    **cfg['train'] 
)

wandb.finish()