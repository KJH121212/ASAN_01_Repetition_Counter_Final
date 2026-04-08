import sys
import json
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))  # 프로젝트 루트 경로 추가
from utils.img_preprocessing import letterbox


# =================================================================
# 🛠️ 유틸리티 함수 모음
# =================================================================
def get_padded_crop(img, bbox, pad_ratio=0.2):
    h_orig, w_orig = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    dw, dh = w * pad_ratio, h * pad_ratio
    nx1, ny1 = max(0, int(x1 - dw)), max(0, int(y1 - dh))
    nx2, ny2 = min(w_orig, int(x2 + dw)), min(h_orig, int(y2 + dh))
    return img[ny1:ny2, nx1:nx2], (nx1, ny1)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)

def scale_coords_back(kpts, ratio, pad, offset):
    kpts[:, 0] -= pad[0]
    kpts[:, 1] -= pad[1]
    kpts[:, :2] /= ratio
    kpts[:, 0] += offset[0]
    kpts[:, 1] += offset[1]
    return kpts

# =================================================================
# 🎨 스켈레톤 드로잉 유틸리티
# =================================================================
# 1. Pre-trained(17kpt) 연결
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

def draw_pt_17kpt(img, kpts, color=(0, 255, 0), thickness=2):
    """Pre-trained 모델(17kpt)용 드로잉 함수 (Conf 0.25 이상)"""
    for p1, p2 in SKELETON_17:
        if p1 < len(kpts) and p2 < len(kpts):
            x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
            x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
            conf1 = kpts[p1][2] if len(kpts[p1]) > 2 else 1.0
            conf2 = kpts[p2][2] if len(kpts[p2]) > 2 else 1.0
            if conf1 > 0.25 and conf2 > 0.25:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    for pt in kpts:
        x, y = int(pt[0]), int(pt[1])
        conf = pt[2] if len(pt) > 2 else 1.0
        if conf > 0.25:
            cv2.circle(img, (x, y), 4, color, -1)
    return img

# 2. 파인튜닝(12kpt) 연결
LINKS_12 = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),  # 상체
    (0, 6), (1, 7), (6, 7),                  # 몸통
    (6, 8), (8, 10), (7, 9), (9, 11)         # 하체
]

def draw_pred_12kpt(img, kpts, color=(0, 255, 255), thickness=2):
    """예측 모델(12kpt)용 드로잉 함수 - Conf 무시"""
    for p1, p2 in LINKS_12:
        if p1 < len(kpts) and p2 < len(kpts):
            x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
            x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
            if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
                cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    for i in range(len(kpts)):
        x, y = int(kpts[i][0]), int(kpts[i][1])
        if x > 0 or y > 0: 
            cv2.circle(img, (x, y), 4, color, -1)
    return img

def calculate_oks_torch(pred_12, gt_12, scale, sigmas):
    dists_sq = torch.sum((pred_12[:, :2] - gt_12[:, :2])**2, dim=1)
    v_mask = gt_12[:, 2] > 0
    if torch.sum(v_mask) == 0: return 0.0
    denom = 2 * (scale**2) * (sigmas**2)
    return (torch.sum(torch.exp(-dists_sq / denom)[v_mask]) / torch.sum(v_mask)).item()

# =================================================================
# 🚀 기능 함수 1. OKS 정량 평가
# =================================================================
def run_oks_evaluation(det_model, pose_model, frame_paths, json_dir, device):
    print("\n" + "="*50)
    print("🚀 [OKS 평가] Single vs Double 정확도 비교 시작")
    print("="*50)
    
    BODY_SIGMAS = torch.tensor([0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089], device=device)
    metrics = {'Single-Stage': {'okss': [], 'det': 0}, 'Double-Stage': {'okss': [], 'det': 0}}
    total_valid_frames = 0
    
    for f_path in tqdm(frame_paths, desc="OKS 계산 중"):
        img = cv2.imread(str(f_path))
        json_path = Path(json_dir) / f"{f_path.stem}.json"
        
        if img is None or not json_path.exists(): continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            if not data['instance_info']: continue
            total_valid_frames += 1
            
            inst = data['instance_info'][0]
            gt_kpts = np.array([[k[0], k[1], s] for k, s in zip(inst['keypoints'], inst['keypoint_scores'])])
            bx1, by1, bx2, by2 = inst['bbox']
            scale = np.sqrt((bx2 - bx1) * (by2 - by1))
            gt_t = torch.tensor(gt_kpts, device=device, dtype=torch.float32)
            
            # Single-Stage
            res_s = pose_model.predict(img, verbose=False, device=device)[0]
            if res_s.keypoints is not None and len(res_s.keypoints.xy) > 0:
                p_t = res_s.keypoints.data[0]
                metrics['Single-Stage']['okss'].append(calculate_oks_torch(p_t, gt_t[5:17], scale, BODY_SIGMAS))
                metrics['Single-Stage']['det'] += 1
                
            # Double-Stage
            res_d = det_model.predict(img, classes=[0], verbose=False, device=device)[0]
            if len(res_d.boxes) > 0:
                box = res_d.boxes[0].xyxy[0].cpu().numpy().astype(int)
                crop, (nx1, ny1) = get_padded_crop(img, box, pad_ratio=0.2)
                if crop.size > 0:
                    c_img, r, p = letterbox(crop, 640)
                    res_p = pose_model.predict(c_img, verbose=False, device=device)[0]
                    if res_p.keypoints is not None and len(res_p.keypoints.xy) > 0:
                        p_t = scale_coords_back(res_p.keypoints.data[0].clone(), r, p, (nx1, ny1))
                        metrics['Double-Stage']['okss'].append(calculate_oks_torch(p_t, gt_t[5:17], scale, BODY_SIGMAS))
                        metrics['Double-Stage']['det'] += 1

    report = pd.DataFrame([{
        "Strategy": k, 
        "Det Rate(%)": round(v['det']/total_valid_frames*100, 2) if total_valid_frames else 0,
        "Mean OKS": round(np.mean(v['okss']), 4) if v['okss'] else 0
    } for k, v in metrics.items()])
    print("\n📊 [포즈 추정 12kpt OKS 비교 리포트]")
    print(report.to_string(index=False))

# =================================================================
# 🚀 기능 함수 2. 단일 이미지 분석
# =================================================================
def process_single_image(det_model, pt_pose_model, ft_pose_model, img_path, out_2x2_path, out_crop_path, device):
    img_path = Path(img_path)
    print(f"\n🚀 [단일 이미지 분석] {img_path.name}")
    img = cv2.imread(str(img_path))
    if img is None: sys.exit("❌ 이미지를 찾을 수 없음.")
    
    h, w = img.shape[:2]
    cell_w, cell_h = w // 2, h // 2
    
    # 1. ORIGIN
    p1 = cv2.resize(img.copy(), (cell_w, cell_h))
    cv2.putText(p1, "1. ORIGIN", (10, 25), 1, 1.2, (255, 255, 255), 2)
    
    # 2. PRE-TRAINED (17kpt)
    res_pt = pt_pose_model.predict(img, verbose=False, device=device)[0]
    p2_canvas = img.copy()
    if res_pt.keypoints is not None and len(res_pt.keypoints.data) > 0:
        # 💡 탐지된 모든 사람을 순회하며 그리기
        for pt_kpts in res_pt.keypoints.data:
            p2_canvas = draw_pt_17kpt(p2_canvas, pt_kpts.cpu().numpy(), color=(0, 255, 0))
    p2 = cv2.resize(p2_canvas, (cell_w, cell_h))
    cv2.putText(p2, "2. PRE-TRAINED (17Kpts)", (10, 25), 1, 1.2, (0, 255, 0), 2)
    
    # 3. SINGLE-STAGE (Fine-tuned)
    res_s = ft_pose_model.predict(img, verbose=False, device=device)[0]
    p3_canvas = img.copy()
    if res_s.keypoints is not None and len(res_s.keypoints.data) > 0:
        # 💡 탐지된 모든 사람을 순회하며 그리기 (xy 대신 data 사용)
        for s_kpts in res_s.keypoints.data:
            p3_canvas = draw_pred_12kpt(p3_canvas, s_kpts.cpu().numpy(), color=(0, 0, 255))
    p3 = cv2.resize(p3_canvas, (cell_w, cell_h))
    cv2.putText(p3, "3. FINE-TUNED (Single)", (10, 25), 1, 1.2, (0, 0, 255), 2)
    
    # 4. DOUBLE-STAGE (Det + Fine-tuned Pose)
    res_d = det_model.predict(img, classes=[0], conf=0.15, verbose=False, device=device)[0]
    p4_canvas = img.copy()
    crop_vis_list = []
    
    if len(res_d.boxes) > 0:
        for i, box_data in enumerate(res_d.boxes):
            box = box_data.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(p4_canvas, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            crop, (nx1, ny1) = get_padded_crop(img, box, pad_ratio=0.2)
            if crop.size == 0: continue
            
            c_img, r, p = letterbox(crop, 640)
            res_p = ft_pose_model.predict(c_img, verbose=False, device=device)[0]
            
            raw_crop_res = cv2.resize(crop, (640, 640))
            input_inspect = c_img.copy()
            
            if res_p.keypoints is not None and len(res_p.keypoints.xy) > 0:
                pkpts = res_p.keypoints.xy[0].cpu().numpy()
                input_inspect = draw_pred_12kpt(input_inspect, pkpts, color=(0, 255, 255), thickness=3)
                pkpts_full = scale_coords_back(res_p.keypoints.xy[0].clone(), r, p, (nx1, ny1)).cpu().numpy()
                p4_canvas = draw_pred_12kpt(p4_canvas, pkpts_full, color=(0, 255, 255))
            else:
                cv2.putText(input_inspect, "FAILED", (50, 320), 1, 2.0, (0, 0, 255), 2)
                
            cv2.putText(raw_crop_res, f"RAW CROP {i}", (10, 30), 1, 1.0, (255, 255, 255), 2)
            cv2.putText(input_inspect, f"MODEL INPUT {i}", (10, 30), 1, 1.0, (0, 255, 255), 2)
            crop_vis_list.append(np.hstack((raw_crop_res, input_inspect)))
            
    p4 = cv2.resize(p4_canvas, (cell_w, cell_h))
    cv2.putText(p4, "4. DOUBLE-STAGE (Multi)", (10, 25), 1, 1.2, (0, 255, 255), 2)
    
    final_report = np.vstack((np.hstack((p1, p2)), np.hstack((p3, p4))))
    cv2.imwrite(str(out_2x2_path), final_report)
    if crop_vis_list: cv2.imwrite(str(out_crop_path), np.vstack(crop_vis_list))
    print(f"✅ 단일 이미지 저장 완료:\n - {out_2x2_path}\n - {out_crop_path}")

# =================================================================
# 🚀 기능 함수 3. 비디오 렌더링 (4분할)
# =================================================================
def process_video(det_model, pt_pose_model, ft_pose_model, frame_paths, out_video_path, device):
    print("\n" + "="*50)
    print("🚀 [비디오 렌더링] 4분할 영상 생성 시작")
    print("="*50)
    
    h, w = cv2.imread(str(frame_paths[0])).shape[:2]
    cell_w, cell_h = w // 2, h // 2
    video_out = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    
    for f_path in tqdm(frame_paths, desc="비디오 프레임 처리 중"):
        img = cv2.imread(str(f_path))
        if img is None: continue
        
        # 1. RAW
        v_raw = cv2.resize(img.copy(), (cell_w, cell_h))
        cv2.putText(v_raw, "1. RAW", (10, 30), 1, 1.2, (255, 255, 255), 2)

        # 2. PRE-TRAINED (17kpt)
        v_pt = img.copy()
        res_pt = pt_pose_model.predict(img, verbose=False, device=device)[0]
        if res_pt.keypoints is not None and len(res_pt.keypoints.data) > 0:
            # 💡 탐지된 모든 사람 그리기
            for pt_kpts in res_pt.keypoints.data:
                v_pt = draw_pt_17kpt(v_pt, pt_kpts.cpu().numpy(), color=(0, 255, 0))
        v_pt = cv2.resize(v_pt, (cell_w, cell_h))
        cv2.putText(v_pt, "2. PRE-TRAINED (17Kpts)", (10, 30), 1, 1.2, (0, 255, 0), 2)

        # 3. SINGLE-STAGE (Fine-tuned)
        v_s = img.copy()
        res_s = ft_pose_model.predict(img, verbose=False, device=device)[0]
        if res_s.keypoints is not None and len(res_s.keypoints.data) > 0:
            # 💡 탐지된 모든 사람 그리기
            for s_kpts in res_s.keypoints.data:
                v_s = draw_pred_12kpt(v_s, s_kpts.cpu().numpy(), color=(0, 0, 255))
        v_s = cv2.resize(v_s, (cell_w, cell_h))
        cv2.putText(v_s, "3. FINE-TUNED (Single)", (10, 30), 1, 1.2, (0, 0, 255), 2)

        # 4. DOUBLE-STAGE (Det + Fine-tuned Pose)
        v_d = img.copy()
        res_d = det_model.predict(img, classes=[0], verbose=False, device=device)[0]
        if len(res_d.boxes) > 0:
            box = res_d.boxes[0].xyxy[0].cpu().numpy().astype(int)
            crop, (nx1, ny1) = get_padded_crop(img, box, pad_ratio=0.2)
            if crop.size > 0:
                c_img, r, p = letterbox(crop, 640)
                res_p = ft_pose_model.predict(c_img, verbose=False, device=device)[0]
                if res_p.keypoints is not None and len(res_p.keypoints.xy) > 0:
                    pkpts = scale_coords_back(res_p.keypoints.xy[0].clone(), r, p, (nx1, ny1))
                    v_d = draw_pred_12kpt(v_d, pkpts.cpu().numpy(), color=(0, 255, 255))
        v_d = cv2.resize(v_d, (cell_w, cell_h))
        cv2.putText(v_d, "4. DOUBLE-STAGE (Multi)", (10, 30), 1, 1.2, (0, 255, 255), 2)

        combined = np.vstack([np.hstack([v_raw, v_pt]), np.hstack([v_s, v_d])])
        video_out.write(combined)

    video_out.release()
    print(f"🎉 비디오 저장 완료: {out_video_path}")

# =================================================================
# ⚙️ MAIN 실행 블록
# =================================================================
if __name__ == "__main__":
    
    # -------------------------------------------------------------
    # 1. 실행 모드 설정
    # -------------------------------------------------------------
    VISUALIZE_MODE = 'image'   # 선택: 'image', 'video', 'none'
    RUN_OKS_EVAL   = False      # 선택: True, False

    # -------------------------------------------------------------
    # 2. 타겟 및 이미지 번호 설정
    # -------------------------------------------------------------
    TARGET_IDX     = 0
    SINGLE_IMG_NUM = 103       

    # -------------------------------------------------------------
    # 3. 디렉토리 및 모델 경로 설정
    # -------------------------------------------------------------
    BASE_DIR       = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
    DATA_DIR       = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    METADATA_PATH  = DATA_DIR / "metadata_v2.1.csv"
    
    DET_MODEL_PATH      = "yolo11n.pt"
    PRE_POSE_MODEL_PATH = "yolo11n-pose.pt"
    POSE_MODEL_PATH     = DATA_DIR / "checkpoints/YOLO_FINETUNING/v2.0_step1/weights/best.pt"

    # -------------------------------------------------------------
    # 4. 출력 파일명 설정 (mkdir 처리 포함)
    # -------------------------------------------------------------
    COMPARE_OUT_DIR = BASE_DIR / "yolo_runner/compare"
    COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True) # 💡 폴더 생성 추가!

    OUT_SINGLE_2x2  = str(COMPARE_OUT_DIR / f"image_{TARGET_IDX}_{SINGLE_IMG_NUM:06d}_2x2.jpg")
    OUT_SINGLE_CROP = str(COMPARE_OUT_DIR / f"image_{TARGET_IDX}_{SINGLE_IMG_NUM:06d}_crops.jpg")
    OUT_VIDEO_MP4   = str(COMPARE_OUT_DIR / f"compare_video_{TARGET_IDX}.mp4")

    # =============================================================
    # [이하 코드는 설정값을 바탕으로 자동으로 동작함]
    # =============================================================
    sys.path.append(str(BASE_DIR))
    from utils.path_list import path_list

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("🔄 모델 로드 중...")
    # 💡 세 가지 모델 모두 실제 YOLO 객체로 초기화!
    det_model = YOLO(DET_MODEL_PATH).to(device)
    pt_pose_model = YOLO(PRE_POSE_MODEL_PATH).to(device)
    ft_pose_model = YOLO(POSE_MODEL_PATH).to(device)
    
    meta_df = pd.read_csv(METADATA_PATH)
    common_path = meta_df.iloc[TARGET_IDX]['common_path']
    paths = path_list(common_path)
    
    frame_dir = Path(paths['frame'])
    json_dir = paths['interp_data']
    frame_paths = sorted(list(frame_dir.glob("*.jpg")))

    if not frame_paths:
        sys.exit(f"❌ {frame_dir} 경로에 프레임 이미지가 없습니다.")

    # 1. OKS 평가 실행
    if RUN_OKS_EVAL:
        run_oks_evaluation(det_model, ft_pose_model, frame_paths, json_dir, device)
        
    # 2. 시각화 실행
    if VISUALIZE_MODE == 'image':
        single_img_path = frame_dir / f"{SINGLE_IMG_NUM:06d}.jpg"
        if not single_img_path.exists():
            print(f"⚠️ {single_img_path.name} 이미지가 없어 첫 프레임으로 대체함.")
            single_img_path = frame_paths[0]
            
        # 💡 객체로 로드된 모델(pt_pose_model, ft_pose_model)들을 명시적으로 전달!
        process_single_image(
            det_model=det_model, 
            pt_pose_model=pt_pose_model,    
            ft_pose_model=ft_pose_model, 
            img_path=single_img_path, 
            out_2x2_path=OUT_SINGLE_2x2, 
            out_crop_path=OUT_SINGLE_CROP, 
            device=device
        )           

    elif VISUALIZE_MODE == 'video':
        # 💡 비디오 렌더링도 Pre-trained와 비교하도록 함수 서명 변경 대응
        process_video(det_model, pt_pose_model, ft_pose_model, frame_paths, OUT_VIDEO_MP4, device)
        
    elif VISUALIZE_MODE == 'none':
        print("✅ 시각화 모드가 'none'이므로 시각화를 건너뜁니다.")