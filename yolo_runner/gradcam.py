# single 이미지에 대해 FT/PT 모델의 Grad-CAM 시각화를 모든 layer에서 2x2 그리드로 생성하는 스크립트
import cv2
import numpy as np
import torch
import sys
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =================================================================
# 1. 경로 및 데이터 로드 설정
# =================================================================
# 사용자 환경에 맞춰 수정바람
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))
from utils.path_list import path_list

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"

meta_df = pd.read_csv(METADATA_PATH)
target = 0
img_num = 195  # 예시 이미지 번호 (실제 데이터에 맞게 수정 필요)
common_path = meta_df.iloc[target]['common_path']
paths = path_list(common_path)
img_path = str(paths['frame'] / f"{img_num:06d}.jpg")

output_dir = Path(f"./grad-cam-{target}-{img_num}")
output_dir.mkdir(parents=True, exist_ok=True)

# =================================================================
# 2. Grad-CAM 필수 클래스 및 래퍼
# =================================================================
class RawScoreTarget:
    """Grad-CAM에 전달할 타겟 점수 클래스"""
    def __call__(self, model_output):
        return model_output

class YOLOPoseWrapper(torch.nn.Module):
    """Skip-connection 및 복합 레이어 출력을 스칼라로 변환하는 래퍼"""
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.feature_output = None
        self.hook_handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # Layer 23(Head) 같은 튜플/리스트 출력 대응
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

# =================================================================
# 3. 드로잉 및 유틸리티 함수
# =================================================================
def plot_custom_12kpt(image, result):
    img_draw = image.copy()
    skeleton_pairs = [
        (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
        (0, 6), (1, 7), (6, 7),
        (6, 8), (8, 10), (7, 9), (9, 11)
    ]
    if result.keypoints is not None and len(result.keypoints) > 0:
        for kpts in result.keypoints.data.cpu().numpy():
            for p1_idx, p2_idx in skeleton_pairs:
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    p1, p2 = kpts[p1_idx], kpts[p2_idx]
                    if (p1[0] > 0 or p1[1] > 0) and (p2[0] > 0 or p2[1] > 0):
                        cv2.line(img_draw, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)
            for i, kpt in enumerate(kpts):
                x, y, _ = kpt
                if x > 0 or y > 0:
                    cv2.circle(img_draw, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(img_draw, str(i), (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return img_draw

def add_label(image, text):
    img_copy = image.copy()
    cv2.rectangle(img_copy, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(img_copy, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return img_copy

def get_layer_description(idx):
    descriptions = {
        0: "Layer 00: Conv - Stem (Initial 3x3 Downsampling)",
        1: "Layer 01: Conv - Backbone Downsampling",
        2: "Layer 02: C3k2 - Backbone Stage 1 (Local Features)",
        3: "Layer 03: Conv - Backbone Downsampling",
        4: "Layer 04: C3k2 - Backbone Stage 2 (Texture/Parts)",
        5: "Layer 05: Conv - Backbone Downsampling",
        6: "Layer 06: C3k2 - Backbone Stage 3 (Object Silhouettes)",
        7: "Layer 07: Conv - Backbone Downsampling",
        8: "Layer 08: C3k2 - Backbone Stage 4 (Complex Semantics)",
        9: "Layer 09: SPPF - Spatial Pyramid Pooling (Multi-scale)",
        10: "Layer 10: C2PSA - Position-Sensitive Attention",
        11: "Layer 11: Upsample - Feature Resolution Recovery",
        12: "Layer 12: Concat - Neck Fusion Stage 1",
        13: "Layer 13: C3k2 - Neck Refining Stage 1",
        14: "Layer 14: Upsample - Feature Resolution Recovery",
        15: "Layer 15: Concat - Neck Fusion Stage 2",
        16: "Layer 16: C3k2 - Neck Refining Stage 2 (Detail focus)",
        17: "Layer 17: Conv - Neck Downsampling",
        18: "Layer 18: Concat - Neck Fusion Stage 3",
        19: "Layer 19: C3k2 - Neck Refining Stage 3",
        20: "Layer 20: Conv - Neck Downsampling",
        21: "Layer 21: Concat - Neck Fusion Stage 4",
        22: "Layer 22: C3k2 - Final Neck Output (Global Context)",
        23: "Layer 23: Pose Head - 12 Kpt Coordinate Regression"
    }
    return descriptions.get(idx, f"Layer {idx:02d}: Intermediate Module")

# =================================================================
# 4. Grad-CAM 추출 핵심 함수
# =================================================================
def get_gradcam_image(yolo_obj, layer_idx, text_label):
    model_layers = yolo_obj.model.model
    
    # 💡 [핵심] 23번 레이어는 내부의 실제 연산 층인 cv2 모듈을 타겟팅함
    if layer_idx == 23:
        try:
            target_layer = model_layers[layer_idx].cv2[0][2]
        except:
            target_layer = model_layers[layer_idx]
    else:
        target_layer = model_layers[layer_idx]
        
    wrapped_model = YOLOPoseWrapper(yolo_obj.model, target_layer)
    try:
        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
        
        # 캡처 검증
        if wrapped_model(input_tensor) is None:
            raise RuntimeError(f"Layer {layer_idx} Bypassed")
            
        grayscale_cam = cam(input_tensor=input_tensor, targets=[RawScoreTarget()])[0, :]
        vis_rgb = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        return add_label(vis_bgr, text_label)
    finally:
        wrapped_model.remove_hook()

# =================================================================
# 5. 메인 실행 및 2x2 그리드 생성
# =================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
yolo_A = YOLO("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/YOLO_FINETUNING/v2.0_step1/weights/best.pt")
yolo_B = YOLO("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/YOLO/yolo11m-pose.pt")

img = cv2.imread(img_path)
img_resized = cv2.resize(img, (640, 640))
rgb_img = np.float32(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)) / 255
input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
input_tensor.requires_grad = True

print(f"🚀 Skip-connection 및 Pose Head 대응 통합 Grad-CAM 시작함.")

# Pose 결과 (Top Row)
res_A = yolo_A.predict(img_resized, verbose=False)[0]
res_B = yolo_B.predict(img_resized, verbose=False)[0]
pose_A = add_label(plot_custom_12kpt(img_resized, res_A), "FT Model [12 Kpts]")
pose_B = add_label(res_B.plot(), "PT Model [17 Kpts]")

v_sep = np.zeros((640, 5, 3), dtype=np.uint8)
h_sep = np.zeros((5, 1285, 3), dtype=np.uint8)
top_row = np.hstack((pose_A, v_sep, pose_B))

for layer_num in range(24):
    try:
        cam_A = get_gradcam_image(yolo_A, layer_num, f"FT Grad L{layer_num:02d}")
        cam_B = get_gradcam_image(yolo_B, layer_num, f"PT Grad L{layer_num:02d}")
        
        bottom_row = np.hstack((cam_A, v_sep, cam_B))
        grid_2x2 = np.vstack((top_row, h_sep, bottom_row))
        
        header = np.full((60, grid_2x2.shape[1], 3), 255, dtype=np.uint8)
        desc = get_layer_description(layer_num)
        cv2.putText(header, desc, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        final_output = np.vstack((header, grid_2x2))
        cv2.imwrite(str(output_dir / f"compare_2x2_L{layer_num:02d}.jpg"), final_output)
        print(f"✅ [Layer {layer_num:02d}] 완료")
        
    except Exception as e:
        print(f"⚠️ [Layer {layer_num:02d}] 실패: {e}")

print(f"🎉 모든 작업이 완료됨. 저장 경로: {output_dir}")