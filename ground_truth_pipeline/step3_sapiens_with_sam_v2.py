import sys
import json
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
try:
    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    from mmpose.structures import PoseDataSample, merge_data_samples
    from mmengine.dataset import Compose
except ImportError as e:
    print(f"[CRITICAL ERROR] mmpose Import 실패: {e}")
    sys.exit(1)

try:
    from utils.boundary_box import extract_bbox_and_id
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.boundary_box import extract_bbox_and_id

def to_py(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

# ============================================================
# 0️⃣ [NEW] BBox 안정화 매니저 (핵심 로직)
# ============================================================
class DynamicBBoxManager:
    def __init__(self, expand_ratio=1.0, min_occupancy=0.7, smoothing=0.2, cooldown=30):
        self.expand_ratio = expand_ratio
        self.margin_threshold = 1.0 - min_occupancy 
        self.alpha = smoothing
        self.current_box = None 
        self.cooldown_max = cooldown
        self.cooldown_timer = 0

    def update(self, mask_bbox):
        if self.current_box is None:
            self.current_box = self._make_target_box(mask_bbox, self.expand_ratio)
            return self.current_box

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        
        if self._is_out_of_bound(mask_bbox, self.current_box):
            target_box = self._make_target_box(mask_bbox, self.expand_ratio)
            self.current_box = self._smooth_update(self.current_box, target_box)
            self.cooldown_timer = self.cooldown_max 
            
        elif self.cooldown_timer == 0 and \
             self._has_excessive_margin(mask_bbox, self.current_box, self.margin_threshold):
            target_box = self._make_target_box(mask_bbox, self.expand_ratio)
            self.current_box = self._smooth_update(self.current_box, target_box)
            self.cooldown_timer = self.cooldown_max
        else:
            pass 

        return self.current_box

    def _make_target_box(self, box, ratio):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        nw, nh = w * ratio, h * ratio
        return [cx - nw/2, cy - nh/2, cx + nw/2, cy + nh/2]

    def _is_out_of_bound(self, mask, view):
        pad = 0.1 
        return (mask[0] < view[0]-pad) or (mask[1] < view[1]-pad) or \
               (mask[2] > view[2]+pad) or (mask[3] > view[3]+pad)

    def _has_excessive_margin(self, mask, view, threshold):
        view_w = view[2] - view[0]
        view_h = view[3] - view[1]
        mask_w = mask[2] - mask[0]
        mask_h = mask[3] - mask[1]
        
        if view_w <= 0 or view_h <= 0: return True
        
        margin_w = 1.0 - (mask_w / view_w)
        margin_h = 1.0 - (mask_h / view_h)
        return (margin_w > threshold) or (margin_h > threshold)

    def _smooth_update(self, current, target):
        return [c * (1 - self.alpha) + t * self.alpha for c, t in zip(current, target)]

def stabilize_bboxes(tasks):
    print("🌊 BBox 안정화 로직 적용 중 (Hysteresis)...")
    managers = {} 
    for sam_file, file_name, objects in tasks:
        for obj in objects:
            if not obj.get('bbox'): continue
            
            oid = obj['id']
            raw_bbox = obj['bbox']
            
            if oid not in managers:
                managers[oid] = DynamicBBoxManager(expand_ratio=1.1, min_occupancy=0.7, smoothing=0.2)
            
            stable_bbox = managers[oid].update(raw_bbox)
            obj['bbox'] = stable_bbox
            
    return tasks

# ============================================================
# 1️⃣ Dataset: Gray Padding (Letterbox) 적용 + Batch 준비
# ============================================================
class SapiensBatchDataset(Dataset):
    def __init__(self, tasks, frame_dir, input_size=(1024, 768)):
        self.frame_dir = Path(frame_dir)
        self.items = []
        self.input_size = input_size 
        
        for sam_file, img_path, rel_path, objects in tasks:
            f_idx = int(sam_file.stem) if sam_file.stem.isdigit() else 0
            for obj in objects:
                if obj.get('bbox'):
                    self.items.append({
                        'rel_path': rel_path,      # 💡 저장할 때 쓸 하위 폴더 포함 경로
                        'img_path': img_path,      # 💡 읽어올 이미지의 정확한 전체 경로
                        'frame_idx': f_idx, 
                        'obj_id': obj['id'], 
                        'bbox': obj['bbox'] 
                    })

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(str(item['img_path'])) # 💡 file_name 조합 대신 정확한 경로 사용
        if img is None: return None
        
        x1, y1, x2, y2 = item['bbox']
        img_h, img_w = img.shape[:2]
        
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2
        
        nx1, ny1 = max(0, int(x1)), max(0, int(y1))
        nx2, ny2 = min(img_w, int(x2)), min(img_h, int(y2))
        
        crop = img[ny1:ny2, nx1:nx2].copy()
        
        if crop.size == 0: return None, None 

        target_w, target_h = self.input_size
        h, w = crop.shape[:2]
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(crop, (new_w, new_h))
        
        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        input_img = canvas.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        input_img = (input_img - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        
        meta = {
            'rel_path': str(item['rel_path']), # 💡 저장 시 구조 유지를 위해 전달
            'frame_idx': item['frame_idx'],
            'obj_id': item['obj_id'],
            'crop_bbox': [nx1, ny1, nx2, ny2],
            'padding': [pad_x, pad_y],
            'scale_factor': scale,
            'input_size': self.input_size
        }
        
        return torch.from_numpy(input_img), meta

def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch: return None
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

# ============================================================
# 2️⃣ Inference: Full Model API 사용 + Batch 처리
# ============================================================
def run_sapiens_batch_inference(frame_dir, sam_dir, output_dir, config_path, ckpt_path, batch_size=8):
    frame_dir, sam_dir, output_dir = Path(frame_dir), Path(sam_dir), Path(output_dir)
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules()
    
    sam_files = sorted(list(sam_dir.rglob("*.json")))
    tasks = []
    print("📂 SAM JSON 스캔 중...")

    for sam_file in tqdm(sam_files):
        _, objects = extract_bbox_and_id(str(sam_file)) # file_name은 쓰지 않습니다.
        
        rel_path = sam_file.relative_to(sam_dir)
        img_path = frame_dir / rel_path.with_suffix('.jpg') 
        
        if img_path.exists():
            tasks.append((sam_file, img_path, rel_path, objects))

    tasks = stabilize_bboxes(tasks) # (주의: stabilize_bboxes 함수 파라미터 언패킹도 4개로 맞춰야함)

    print("🚀 Sapiens 모델 로드 (Batch Mode)...")
    model = init_model(str(config_path), str(ckpt_path), device='cuda:0')
    model.eval()

    dataset = SapiensBatchDataset(tasks, frame_dir, input_size=(1024, 768))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    print(f"⚡ Batch Inference 시작 (Total Objects: {len(dataset)})")
    
    for batch in tqdm(loader, desc="Processing"):
        if batch is None: continue
        inputs, metas = batch
        inputs = inputs.to('cuda', non_blocking=True)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = model.extract_feat(inputs)
            
            batch_data_samples = [
                PoseDataSample(metainfo=dict(input_size=m['input_size'])) 
                for m in metas
            ]
            preds = model.head.predict(feats, batch_data_samples)

        for i, pred_sample in enumerate(preds):
            meta = metas[i]
            
            if hasattr(pred_sample, 'pred_instances'):
                instances = pred_sample.pred_instances
            else:
                instances = pred_sample
            
            kpts_crop = instances.keypoints
            scores = instances.keypoint_scores
            
            if kpts_crop.ndim == 3:
                kpts_crop = kpts_crop[0]
                scores = scores[0]
            
            pad_x, pad_y = meta['padding']
            scale = meta['scale_factor']
            off_x, off_y = meta['crop_bbox'][:2]
            
            final_kpts = []
            if isinstance(kpts_crop, torch.Tensor): kpts_crop = kpts_crop.cpu().numpy()
            if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()

            for (cx, cy), score in zip(kpts_crop, scores):
                x_nopad = cx - pad_x
                y_nopad = cy - pad_y
                fx = (x_nopad / scale) + off_x
                fy = (y_nopad / scale) + off_y
                final_kpts.append([float(fx), float(fy)]) 
            
            crop_bbox = [float(v) for v in meta['crop_bbox']]
            
            # ⭐️ [핵심 수정]: 000001.json과 동일한 구조 및 키 적용
            instance_item = {
                "keypoints": final_kpts,
                "keypoint_scores": scores.tolist(),
                "bbox": crop_bbox,
                "bbox_score": 1.0,               # 샘플과 동일하게 추가
                "instance_id": int(meta['obj_id']) # 샘플과 동일하게 원복
            }
            
            save_path = output_dir / meta['rel_path']  
            save_path.parent.mkdir(parents=True, exist_ok=True) # 💡 폴더 없으면 생성       

            if save_path.exists():
                with open(save_path, "r", encoding="utf-8") as f:
                    data_j = json.load(f)
                data_j['instance_info'].append(instance_item)
            else:
                # ⭐️ [핵심 수정]: 최상단에 file_name 제거, 샘플과 동일하게 유지
                data_j = {
                    "frame_index": meta['frame_idx'],
                    "meta_info": to_py(model.dataset_meta),
                    "instance_info": [instance_item]
                }
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data_j, f, ensure_ascii=False, indent=4) # indent=4로 예쁘게 출력
                
    return len(list(output_dir.glob("*.json")))

#=============================
# patient 누락 파일 목록만 대상으로 sam_2_sapiens 실행
#==============================
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# 1. 사용자님이 검증하신 누락 데이터 분석 함수 (그대로 유지)
def find_missing_instances(sam_dir, kpt_dir, target_id):
    sam_dir, kpt_dir = Path(sam_dir), Path(kpt_dir)
    target_id = int(target_id)
    needs_processing = []
    
    # 기준을 kpt_dir에 있는 파일들로 잡거나 sam_dir로 잡을 수 있는데, 
    # 매칭 시도를 했던 sam_files 기준으로 루프를 도는 것이 안전합니다.
    sam_files = sorted(list(sam_dir.glob("*.json")))
    
    for sam_path in tqdm(sam_files, desc="🔍 누락 데이터 분석 중"):
        try:
            rel_path = sam_path.relative_to(sam_dir) # 💡 01/000000.json
            res_path = kpt_dir / rel_path # 💡 동일한 구조의 kpt_dir 위치
            target_found_in_kpt = False

            # 1. Skeleton(결과) 파일 먼저 확인
            if res_path.exists():
                with open(res_path, "r", encoding="utf-8") as f:
                    res_data = json.load(f)
                
                kpt_ids = [
                    int(inst['instance_id']) for inst in res_data.get('instance_info', []) 
                    if inst and inst.get('instance_id') is not None
                ]
                
                if target_id in kpt_ids:
                    target_found_in_kpt = True

            # 2. 결과에 없을 경우에만 SAM 파일 확인
            if not target_found_in_kpt:
                with open(sam_path, "r", encoding="utf-8") as f:
                    sam_data = json.load(f)
                
                sam_ids = [int(obj['id']) for obj in sam_data.get('objects', []) if obj.get('id') is not None]
                
                # Skeleton에는 없는데 SAM에는 존재한다면? -> "매칭 실패 누락"
                if target_id in sam_ids:
                    needs_processing.append(rel_path)
                # 둘 다 없다면? -> "리스트 추가 안 함" (자연스럽게 넘어감)
                
        except Exception as e:
            print(f"\n⚠️ 오류 발생 [{sam_path.name}]: {e}")

    print(f"\n📊 검사 완료: 총 {len(needs_processing)}개의 매칭 실패 프레임을 발견했습니다.")
    return needs_processing


# 2. 추출된 리스트만 가지고 추론하는 함수
def run_sapiens_inference_from_list(missing_list, frame_dir, sam_dir, output_dir, config_path, ckpt_path, target_id, batch_size=8):
    frame_dir, sam_dir, output_dir = Path(frame_dir), Path(sam_dir), Path(output_dir)
    target_id = int(target_id)

    # 작업 구성
    tasks = []
    print(f"📦 작업 리스트 구성 중...")
    for rel_path in missing_list: # 💡 file_name 대신 rel_path를 받음
        sam_path = sam_dir / rel_path
        with open(sam_path, "r", encoding="utf-8") as f:
            sam_data = json.load(f)
        
        target_objs = [obj for obj in sam_data.get('objects', []) if int(obj['id']) == target_id]
        
        # 💡 프레임 이미지 경로도 하위 폴더 구조를 따라가도록 설정
        img_path = frame_dir / rel_path.with_suffix('.jpg')
        
        if img_path.exists():
            tasks.append((sam_path, img_path, rel_path, target_objs)) # 💡 통일된 형식으로 전달

    if not tasks:
        print("✅ 처리할 작업이 없습니다.")
        return

    # BBox 안정화 (기존 함수 활용)
    tasks = stabilize_bboxes(tasks)

    # 모델 로드
    print(f"🚀 Sapiens 모델 로딩: {ckpt_path}")
    model = init_model(str(config_path), str(ckpt_path), device='cuda:0')
    model.eval()

    dataset = SapiensBatchDataset(tasks, frame_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)

    # 추론 루프
    for batch in tqdm(loader, desc="⚡ 추론 진행 중"):
        if batch is None: continue
        inputs, metas = batch
        inputs = inputs.to('cuda', non_blocking=True)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = model.extract_feat(inputs)
            preds = model.head.predict(feats, [PoseDataSample(metainfo=dict(input_size=m['input_size'])) for m in metas])

        for i, pred_sample in enumerate(preds):
            meta = metas[i]
            
            # [에러 수정]: AttributeError 방어 로직
            # pred_sample이 InstanceData일 경우 바로 접근, 아니면 pred_instances 접근
            if hasattr(pred_sample, 'pred_instances'):
                instances = pred_sample.pred_instances
            else:
                instances = pred_sample 
            
            kpts_crop = instances.keypoints
            scores = instances.keypoint_scores

            # 차원 정리 (N, K, 2) -> (K, 2)
            if kpts_crop.ndim == 3:
                kpts_crop = kpts_crop[0]
                scores = scores[0]
            
            if isinstance(kpts_crop, torch.Tensor): kpts_crop = kpts_crop.cpu().numpy()
            if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()

            pad_x, pad_y = meta['padding']
            scale = meta['scale_factor']
            off_x, off_y = meta['crop_bbox'][:2]
            
            final_kpts = []
            for (cx, cy) in kpts_crop:
                fx = ((cx - pad_x) / scale) + off_x
                fy = ((cy - pad_y) / scale) + off_y
                final_kpts.append([float(fx), float(fy)])

            instance_item = {
                "keypoints": final_kpts,
                "keypoint_scores": scores.tolist(),
                "bbox": [float(v) for v in meta['crop_bbox']],
                "bbox_score": 1.0,
                "instance_id": target_id
            }

            # 파일 업데이트
            save_path = output_dir / meta['rel_path']
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.exists():
                with open(save_path, "r", encoding="utf-8") as f:
                    data_j = json.load(f)
                # 중복 추가 방지
                existing_ids = [inst.get('instance_id') for inst in data_j.get('instance_info', [])]
                if target_id not in existing_ids:
                    data_j['instance_info'].append(instance_item)
            else:
                data_j = {
                    "frame_index": meta['frame_idx'],
                    "meta_info": to_py(model.dataset_meta),
                    "instance_info": [instance_item]
                }
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data_j, f, ensure_ascii=False, indent=4)