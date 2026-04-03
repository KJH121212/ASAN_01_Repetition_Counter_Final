import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

# ---------------------------------------------------------
# 1. 환경 설정 및 데이터 준비
# ---------------------------------------------------------
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

CONFIG_PATH = BASE_DIR/"configs/yolo/exp_v2.0_step1_m.yaml"
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# ---------------------------------------------------------
# 🛠️ 함수: 경로 유동성 해결 & 메타데이터 추출
# ---------------------------------------------------------
def update_data_yaml_and_get_info(yaml_path):
    path_obj = Path(yaml_path)
    with open(path_obj, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    current_data_dir = path_obj.parent.resolve()
    print(f"🔄 데이터 경로 갱신: {data_cfg.get('path')} -> {current_data_dir}")
    data_cfg['path'] = str(current_data_dir)
    
    step_info = data_cfg.get('sampling_step', 'Unknown')
    print(f"ℹ️ 데이터셋 Sampling Step: {step_info}")

    with open(path_obj, 'w') as f:
        yaml.dump(data_cfg, f, sort_keys=False)
    return str(path_obj), step_info

target_data_yaml = cfg['data']['config_path']
fixed_data_yaml, dataset_step = update_data_yaml_and_get_info(target_data_yaml)

# ---------------------------------------------------------
# 🚀 WandB 초기화
# ---------------------------------------------------------
PROJECT = cfg['project_name']
RUN_NAME = cfg['run_name']

if cfg['logging']['use_wandb'] and WANDB_API_KEY:
    try:
        wandb.login(key=WANDB_API_KEY)
        wandb_config = cfg.copy()
        wandb_config['dataset'] = {'sampling_step': dataset_step, 'yaml_path': fixed_data_yaml}

        wandb.init(
            project=PROJECT,
            name=RUN_NAME,
            config=wandb_config,
            resume="allow",
            dir=cfg['output']['base_dir']
        )
        print(f"✅ WandB 초기화 성공 (Pose Task | Step: {dataset_step})")
    except Exception as e:
        print(f"⚠️ WandB 초기화 실패: {e}")

# ---------------------------------------------------------
# 🛠️ [핵심] 커스텀 WandB 콜백 (Pose Metrics 포함)
# ---------------------------------------------------------
def on_train_epoch_end(trainer):
    """
    매 에폭 종료 시 실행. Pose Loss, Box Loss, mAP 등을 모두 기록합니다.
    """
    if wandb.run:
        # trainer.metrics 안에는 'pose_loss', 'box_loss' 등이 자동으로 포함됩니다.
        wandb.log(trainer.metrics)
        wandb.log({"epoch": trainer.epoch + 1})

# ---------------------------------------------------------
# 🤖 스마트 모델 로드 & 이어하기 (Pose Model 전용)
# ---------------------------------------------------------
CHECKPOINT_DIR = os.path.join(cfg['output']['base_dir'], RUN_NAME, 'weights')
LAST_PT_PATH = os.path.join(CHECKPOINT_DIR, 'last.pt')
BASE_MODEL_PATH = str(cfg['model']['base_path'])

print(BASE_MODEL_PATH)
resume_status = False

# 1. 이어하기 (last.pt) 체크
if os.path.exists(LAST_PT_PATH):
    print(f"🔄 [Resume] 이전 학습 체크포인트를 발견했습니다: {LAST_PT_PATH}")
    model = YOLO(LAST_PT_PATH)
    resume_status = True

# 2. 처음 시작 (설정 파일의 모델 경로 사용)
elif os.path.exists(BASE_MODEL_PATH):
    print(f"🆕 [Start] 설정된 Pose 모델을 로드합니다: {BASE_MODEL_PATH}")
    model = YOLO(BASE_MODEL_PATH)
    resume_status = False

# 3. 파일 없음 (자동 다운로드 - Pose 버전 명시)
else:
    print(f"⚠️ [Download] 모델 파일을 찾을 수 없어 'yolo11m-pose.pt'를 다운로드합니다.")
    # 사용자가 실수로 일반 모델을 적었더라도, 파일이 없으면 확실하게 pose 모델을 받도록 처리
    model = YOLO("yolo11m-pose.pt") 
    resume_status = False

# ---------------------------------------------------------
# 🔗 콜백 등록 및 학습 시작
# ---------------------------------------------------------
# 커스텀 콜백 등록
model.add_callback("on_train_epoch_end", on_train_epoch_end)
print("✅ 커스텀 WandB 콜백 등록 완료")

print(f"\n🔥 Pose Estimation 학습 시작: {RUN_NAME} (Resume: {resume_status})")

model.train(
    data=fixed_data_yaml,
    project=cfg['output']['base_dir'], 
    name=RUN_NAME,
    resume=resume_status,
    plots=True,
    **cfg['train'] 
)

if wandb.run:
    wandb.finish()