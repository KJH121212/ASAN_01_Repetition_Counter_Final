import sys
import random
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =================================================================
# ⚙️ 1. 경로 및 샘플링 설정
# =================================================================
NUM_SAMPLES = 10  # 랜덤으로 추출할 총 이미지 개수

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))
from utils.path_list import path_list
from utils.img_preprocessing import letterbox

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"

# 모델 경로
PT_POSE_PATH = "yolo11n-pose.pt" 
FT_POSE_PATH = DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step1/weights/best.pt"

# 메인 출력 폴더
MAIN_OUT_DIR = Path("./grad_cam_head_2x2")

# =================================================================
# 🛠️ 2. Grad-CAM 필수 클래스 및 래퍼
# =================================================================
class RawScoreTarget:
    def __call__(self, model_output):
        return model_output

class YOLOPoseWrapper(torch.nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.feature_output = None
        self.hook_handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        if isinstance(output, (list, tuple)):
            self.feature_output = output[0]
        else:
            self.feature_output = output

    def forward(self, x):
        self.feature_output = None
        try:
            self.model(x)
        except Exception:
            pass
        
        if self.feature_output is None:
            return None
        return self.feature_output.mean().reshape(1)
        
    def remove_hook(self):
        self.hook_handle.remove()

def get_head_target_layer(yolo_model, module_name):
    head = yolo_model.model.model[23]
    target_module = getattr(head, module_name)
    
    # cv4도 cv3처럼 한 단계 전 레이어를 보는 것이 훨씬 정확합니다.
    if module_name in ['cv3', 'cv4']:
        return target_module[0][-2] 
    return target_module[0][-1]

def generate_gradcam_image(yolo_obj, img_tensor, rgb_img, module_name, text_label):
    target_layer = get_head_target_layer(yolo_obj, module_name)
    wrapped_model = YOLOPoseWrapper(yolo_obj.model, target_layer)
    
    try:
        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
        if wrapped_model(img_tensor) is None:
            raise RuntimeError(f"{module_name} Bypassed")
            
        grayscale_cam = cam(input_tensor=img_tensor, targets=[RawScoreTarget()])[0, :]
        vis_rgb = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        return add_label(vis_bgr, text_label)
    finally:
        wrapped_model.remove_hook()

# =================================================================
# 🎨 3. 드로잉 및 유틸리티 함수 (모든 사람 그리기 적용)
# =================================================================
def add_label(image, text):
    img_copy = image.copy()
    cv2.rectangle(img_copy, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(img_copy, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return img_copy

# 12Kpts (Fine-Tuned 모델용) - Conf 점수 무시하고 다 그림
LINKS_12 = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11)]

def draw_all_12kpt(img, result):
    img_draw = img.copy()
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        # 💡 탐지된 모든 사람을 순회
        for kpts in result.keypoints.data.cpu().numpy():
            for p1, p2 in LINKS_12:
                if p1 < len(kpts) and p2 < len(kpts):
                    x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
                    x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
                    if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
                        cv2.line(img_draw, (x1, y1), (x2, y2), (255, 255, 0), 2, cv2.LINE_AA)
            for x, y, _ in kpts:
                if x > 0 or y > 0:
                    cv2.circle(img_draw, (int(x), int(y)), 5, (0, 0, 255), -1)
    return img_draw

# 17Kpts (Pre-Trained 모델용) - Conf 0.25 이상만 그림
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

def draw_all_17kpt(img, result):
    img_draw = img.copy()
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        # 💡 탐지된 모든 사람을 순회
        for kpts in result.keypoints.data.cpu().numpy():
            for p1, p2 in SKELETON_17:
                if p1 < len(kpts) and p2 < len(kpts):
                    x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
                    x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
                    conf1 = kpts[p1][2] if len(kpts[p1]) > 2 else 1.0
                    conf2 = kpts[p2][2] if len(kpts[p2]) > 2 else 1.0
                    if conf1 > 0.25 and conf2 > 0.25:
                        cv2.line(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
            for pt in kpts:
                x, y = int(pt[0]), int(pt[1])
                conf = pt[2] if len(pt) > 2 else 1.0
                if conf > 0.25:
                    cv2.circle(img_draw, (x, y), 4, (0, 255, 0), -1)
    return img_draw

# =================================================================
# ⚙️ 4. 2x2 통합 프로세서
# =================================================================
def process_gradcam_2x2(yolo_FT, yolo_PT, img_path, out_dir, target_idx, frame_num, device):
    img = cv2.imread(str(img_path))
    if img is None: return False

    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))    
    rgb_img = np.float32(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)) / 255
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # 스켈레톤 추론 (Top Row)
    res_FT = yolo_FT.predict(img_resized, verbose=False, device=device)[0]
    res_PT = yolo_PT.predict(img_resized, verbose=False, device=device)[0]
    
    pose_FT = add_label(draw_all_12kpt(img_resized, res_FT), "FT Model [12 Kpts]")
    pose_PT = add_label(draw_all_17kpt(img_resized, res_PT), "PT Model [17 Kpts]")
    
    v_sep = np.zeros((640, 5, 3), dtype=np.uint8)
    h_sep = np.zeros((5, 1285, 3), dtype=np.uint8)
    top_row = np.hstack((pose_FT, v_sep, pose_PT))

    # 모듈별 Grad-CAM 생성 (Bottom Row)
    modules = ['cv2', 'cv3', 'cv4']
    for mod in modules:
        try:
            cam_FT = generate_gradcam_image(yolo_FT, input_tensor, rgb_img, mod, f"FT Grad-CAM [{mod.upper()}]")
            cam_PT = generate_gradcam_image(yolo_PT, input_tensor, rgb_img, mod, f"PT Grad-CAM [{mod.upper()}]")
            
            bottom_row = np.hstack((cam_FT, v_sep, cam_PT))
            grid_2x2 = np.vstack((top_row, h_sep, bottom_row))
            
            header = np.full((60, grid_2x2.shape[1], 3), 255, dtype=np.uint8)
            desc_map = {'cv2': 'BBox', 'cv3': 'Class', 'cv4': 'Keypoint'}
            desc = f"Pose Head [{mod.upper()}: {desc_map[mod]}] | Target: {target_idx} | Frame: {frame_num}"
            cv2.putText(header, desc, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            final_output = np.vstack((header, grid_2x2))
            
            # 폴더 생성 및 저장
            mod_dir = out_dir / mod
            mod_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mod_dir / f"T{target_idx}_{frame_num}.jpg"), final_output)
        except Exception as e:
            pass # 캡처 실패 시 무시

    return True

# =================================================================
# 🚀 5. 메인 실행부
# =================================================================
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("🔄 모델 및 메타데이터 로드 중...")
    
    yolo_FT = YOLO(FT_POSE_PATH).to(device)
    yolo_PT = YOLO(PT_POSE_PATH).to(device)
    meta_df = pd.read_csv(METADATA_PATH)

    print(f"\n🚀 랜덤 샘플링 {NUM_SAMPLES}개 2x2 Grad-CAM 분석 시작...")
    
    success_count = 0
    attempts = 0
    
    # --- 1. 랜덤 추출 루프 ---
    with tqdm(total=NUM_SAMPLES, desc="Grad-CAM 2x2 생성") as pbar:
        while success_count < NUM_SAMPLES and attempts < 1000:
            attempts += 1
            
            rand_target_idx = random.randint(0, len(meta_df) - 1)
            common_path = meta_df.iloc[rand_target_idx]['common_path']
            paths = path_list(common_path)
            
            frame_dir = Path(paths['frame'])
            if not frame_dir.exists(): continue
            frame_paths = list(frame_dir.glob("*.jpg"))
            if not frame_paths: continue
                
            rand_frame_path = random.choice(frame_paths)
            
            if process_gradcam_2x2(yolo_FT, yolo_PT, rand_frame_path, MAIN_OUT_DIR, f"{rand_target_idx:03d}", rand_frame_path.stem, device):
                success_count += 1
                pbar.update(1)

    # --- 💡 2. 개별 테스트 이미지 단일 실행 (추가됨) ---
    test_img = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/test/test.jpg")
    if test_img.exists():
        print(f"\n🚀 [추가 테스트] 지정된 단일 이미지 처리 중: {test_img.name}")
        process_gradcam_2x2(yolo_FT, yolo_PT, test_img, MAIN_OUT_DIR, "TEST", "test", device)
        print("✅ 테스트 이미지 저장 완료.")
    else:
        print(f"\n⚠️ 테스트 이미지({test_img})를 찾을 수 없어 건너뜁니다.")

    print(f"\n🎉 모든 처리가 완료되었습니다! 결과물 경로: {MAIN_OUT_DIR.absolute()}")