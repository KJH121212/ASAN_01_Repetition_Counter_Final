from pathlib import Path # 파일 경로 처리를 위해 Path 객체를 가져옵니다.
import json # JSON 파일 읽기/쓰기를 위해 json 모듈을 가져옵니다.
from tqdm import tqdm # 진행 상황을 시각적으로 보여주기 위해 tqdm을 가져옵니다.
from typing import List, Optional, Union # 타입 힌팅을 통해 코드 가독성과 안정성을 높입니다.
import numpy as np # 수학 연산 및 배열 처리를 위해 numpy를 가져옵니다.
from utils.extract_kpt import extract_id_keypoints, normalize_skeleton_array
import pandas as pd # 선형 보간을 위해 pandas를 가져옵니다.

def filter_and_save_json_segment(
    src_kpt_dir: Union[str, Path], 
    dst_kpt_dir: Union[str, Path], 
    target_ids: Optional[List[int]] = None, 
    start_frame: int = 0, 
    end_frame: int = float('inf')
):
    """
    JSON 파일 내 특정 ID만 남기고 나머지는 저장하지 않는 함수
    start_frame과 end_frame을 이용해 특정 구간의 JSON 파일만 처리할 수 있습니다.
    
    Args:
        src_kpt_dir: 원본 JSON 폴더 경로
        dst_kpt_dir: 저장할 JSON 폴더 경로
        target_ids: 남길 ID 리스트. None이면 모든 ID 유지. (예: [1, 2])
        start_frame: 시작 프레임 번호 (파일명 기준)
        end_frame: 종료 프레임 번호 (파일명 기준)
    """
    src_path = Path(src_kpt_dir)
    dst_path = Path(dst_kpt_dir)
    
    # 1. 경로 확인 및 생성
    if not src_path.exists():
        print(f"❌ 원본 경로가 없습니다: {src_path}")
        return False
    
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 2. 파일 리스트 확보 및 숫자 기준 정렬
    # 파일명이 '000123.json' 형태라고 가정하고 정렬
    all_files = sorted(list(src_path.glob("*.json")))
    
    if not all_files:
        print("⚠️ 처리할 JSON 파일이 없습니다.")
        return False

    # 3. 구간(Segment)에 맞는 파일만 선별
    target_files = []
    for f in all_files:
        try:
            # 파일명에서 숫자 추출 (예: '000123.json' -> 123)
            frame_idx = int(f.stem)
            if start_frame <= frame_idx <= end_frame:
                target_files.append(f)
        except ValueError:
            # 파일명이 숫자가 아닌 경우 건너뜀 (혹은 필요시 포함)
            continue
            
    if not target_files:
        print(f"⚠️ 설정한 구간({start_frame}~{end_frame})에 해당하는 파일이 없습니다.")
        return False

    # 4. 필터링 모드 확인 (ID 지정 vs 전체)
    filter_mode = "Specific IDs" if target_ids else "Keep All IDs"
    print(f"\n🚀 Processing Segment: {start_frame} ~ {end_frame}")
    print(f"⚙️ Filter Mode: {filter_mode} (Targets: {target_ids})")
    
    processed_count = 0
    
    # 5. 파일 처리 루프
    for json_file in tqdm(target_files, desc="Processing JSONs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # --- [핵심 로직] ID 필터링 ---
            # target_ids가 존재할 때만 필터링 수행, 없으면 원본 유지
            if target_ids is not None and 'instance_info' in data:
                filtered_instances = [
                    inst for inst in data['instance_info'] 
                    if inst.get('instance_id') in target_ids # 혹은 inst.get('id')
                ]
                data['instance_info'] = filtered_instances
            # ---------------------------

            # 데이터 저장 (파일명 그대로 유지)
            save_path = dst_path / json_file.name
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            processed_count += 1

        except Exception as e:
            print(f"⚠️ Error processing {json_file.name}: {e}")

    print(f"✅ 완료: {processed_count}개의 파일이 '{dst_path}'에 저장되었습니다.\n")
    return True
import numpy as np # 수치 연산 및 배열 처리를 위한 넘파이입니다.
import numpy as np

class PredictiveKalmanFilter:
    def __init__(self, threshold=0.1, q_std=0.05, r_std=0.01):
        self.threshold = threshold
        self.dt = 1.0
        
        self.x = None
        self.P = np.eye(4)
        self.last_valid_z = None # 🌟 [추가] 가장 최근의 정상 측정값을 기억할 변수입니다.
        
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * (q_std**2)
        self.R = np.eye(2) * (r_std**2)

    def process(self, z):
        z_flat = np.array(z).flatten() 
        z_col = np.array(z).reshape(2, 1) 
        
        # 0. 첫 프레임 초기화
        if self.x is None:
            self.x = np.zeros((4, 1)) 
            self.x[:2] = z_col 
            self.last_valid_z = z_flat # 🌟 첫 데이터가 들어오면 무조건 정상 값으로 기억합니다.
            return z_flat

        # 1. 다음 값 유추 (Predict)
        x_pred = self.F @ self.x 
        P_pred = self.F @ self.P @ self.F.T + self.Q 
        pos_pred = (self.H @ x_pred).flatten() 
        
        # 2. 예측값과 실제 입력값의 차이 계산
        dist = np.linalg.norm(z_flat - pos_pred) 
        
        # 3. 차이에 따른 판단 로직
        if dist > self.threshold:
            # 🚨 [Outlier 발생] 예측값 대신 이전 정상 측정값을 반환합니다!
            
            # 다음 프레임에서 필터가 엉뚱한 곳으로 계속 날아가지 않도록, 
            # 내부 상태(위치)도 이전 값으로 고정하고 속도(가속도)는 0으로 멈춰둡니다.
            self.x[:2] = self.last_valid_z.reshape(2, 1)
            self.x[2:] = 0 # X, Y 속도를 0으로 초기화
            self.P = P_pred
            
            return self.last_valid_z # 🌟 예측값이 아닌 '이전 정상 측정값'을 그대로 반환!
            
        else:
            # 🟢 [정상 값] 
            y = z_col - (self.H @ x_pred) 
            S = self.H @ P_pred @ self.H.T + self.R 
            K = P_pred @ self.H.T @ np.linalg.inv(S) 
            
            self.x = x_pred + K @ y 
            self.P = (np.eye(4) - K @ self.H) @ P_pred 
            
            self.last_valid_z = z_flat # 🌟 정상적인 값이므로 "최근 정상값"을 업데이트합니다.
            return z_flat

def apply_custom_kalman(data_np, threshold=0.1, q_std=0.05, r_std=0.01):
    """
    (N, 12, 3) 배열에 커스텀 칼만 필터를 적용합니다.
    """
    frames, n_kpts, _ = data_np.shape
    filtered_np = data_np.copy()
    
    # 각 관절마다 필터 생성
    filters = [PredictiveKalmanFilter(threshold=threshold, q_std=q_std, r_std=r_std) for _ in range(n_kpts)]
    
    for f in range(frames):
        for k in range(n_kpts):
            z = data_np[f, k, :2]
            score = data_np[f, k, 2]
            
            # 관절이 인식된 경우 처리
            if score > 0 and not np.all(z == 0):
                filtered_np[f, k, :2] = filters[k].process(z)
            else:
                # 🌟 [수정됨] 화면에서 가려져 관절을 놓친 경우(결측치)에도, 
                # 예측 보간 대신 '이전 정상 측정값'을 그대로 복사하여 제자리에 멈춰있게 합니다.
                if filters[k].last_valid_z is not None:
                    # 내부 상태도 속도 0으로 묶어둡니다.
                    filters[k].x[:2] = filters[k].last_valid_z.reshape(2, 1)
                    filters[k].x[2:] = 0 
                    
                    filtered_np[f, k, :2] = filters[k].last_valid_z
                    
    return filtered_np

def apply_interpolation_outlier_filter(data_np, max_pixel_speed=50.0, use_iqr=True, iqr_multiplier=3.0):
    """
    속도 및 IQR 기반으로 Outlier를 탐지하여 빈칸(NaN)으로 만든 뒤, 
    앞뒤의 정상값을 선형 보간(Linear Interpolation)으로 부드럽게 이어주는 함수입니다.
    """
    frames, n_kpts, _ = data_np.shape # 데이터의 전체 프레임 수와 관절 개수를 파악합니다.
    filtered_np = data_np.copy() # 원본 훼손을 막기 위해 복사본을 만듭니다.
    
    for k in range(n_kpts): # 모든 관절을 순회합니다.
        
        # ==========================================================
        # 1단계: IQR 기반 속도 임계값(Threshold) 계산
        # ==========================================================
        coords = data_np[:, k, :2] # (Frames, 2) 형태의 현재 관절 x, y 좌표입니다.
        velocity = np.zeros(frames)
        velocity[1:] = np.linalg.norm(coords[1:] - coords[:-1], axis=1) # 프레임 간 이동 거리(속도)를 구합니다.
        
        joint_threshold = max_pixel_speed # 기본 임계값을 설정합니다.
        
        if use_iqr:
            # 완전히 튀는 값이나 0을 제외한 '정상적인 움직임' 속도만 모아봅니다.
            valid_v = velocity[(velocity > 0) & (velocity <= max_pixel_speed)]
            
            if len(valid_v) > 3: # 통계를 낼 수 있는 최소한의 데이터가 있다면
                q1 = np.percentile(valid_v, 25) # 하위 25% 지점의 속도
                q3 = np.percentile(valid_v, 75) # 상위 25% 지점의 속도
                iqr = q3 - q1 # 정상적인 움직임의 변동폭(IQR)입니다.
                
                # 상위 25% 속도보다 '변동폭의 3배' 이상 비정상적으로 빠르면 커트라인으로 잡습니다.
                stat_threshold = q3 + (iqr_multiplier * iqr) 
                joint_threshold = min(max_pixel_speed, stat_threshold) # 둘 중 더 깐깐한 값을 최종 임계값으로 사용합니다.
                
        # ==========================================================
        # 2단계: Outlier 탐지 및 빈칸(NaN) 뚫기
        # ==========================================================
        last_valid_pos = None # '가장 최근에 확인된 정상 좌표'를 기억할 변수입니다.
        
        for f in range(frames):
            z = filtered_np[f, k, :2]
            score = filtered_np[f, k, 2]
            
            # 신뢰도가 0이거나 관절이 (0,0)으로 아예 인식 안 된 경우도 보간 대상(NaN)으로 만듭니다.
            if score == 0 or np.all(z == 0):
                filtered_np[f, k, :2] = np.nan
                continue
                
            if last_valid_pos is None:
                last_valid_pos = z.copy() # 가장 첫 데이터는 정상으로 믿고 시작합니다.
            else:
                dist = np.linalg.norm(z - last_valid_pos) # '마지막 정상 위치'와의 거리를 계산합니다.
                
                if dist > joint_threshold:
                    # 🚨 [Outlier 발생] 거리가 너무 멀면 빈칸(NaN)으로 지워버립니다!
                    filtered_np[f, k, :2] = np.nan 
                else:
                    # 🟢 [정상 값] 정상 궤도라면 '마지막 정상 위치'를 지금 위치로 갱신합니다.
                    last_valid_pos = z.copy()
                    
        # ==========================================================
        # 3단계: 선형 보간법으로 빈칸(NaN) 채우기
        # ==========================================================
        # X좌표와 Y좌표를 각각 Pandas의 Series로 만들어 보간법을 적용합니다.
        series_x = pd.Series(filtered_np[:, k, 0])
        series_y = pd.Series(filtered_np[:, k, 1])
        
        # 앞쪽 정상값과 뒤쪽 정상값을 직선으로 이어(Linear) 비율에 맞춰 빈칸을 채웁니다.
        filtered_np[:, k, 0] = series_x.interpolate(method='linear', limit_direction='both').to_numpy()
        filtered_np[:, k, 1] = series_y.interpolate(method='linear', limit_direction='both').to_numpy()
        
    return filtered_np # Outlier가 부드럽게 교정된 최종 배열을 반환합니다.
import os
import json
import numpy as np
import pandas as pd

def apply_sam_mask_outlier_filter(data_np, sam_dir, patient_id):
    """
    특정 patient_id의 마스크 내부에 키포인트가 위치할 경우, 
    해당 좌표를 제거(NaN)하고 선형 보간합니다.
    """
    frames, n_kpts, _ = data_np.shape
    filtered_np = data_np.copy()
    
    json_files = sorted([f for f in os.listdir(sam_dir) if f.endswith('.json')])
    
    for f_idx in range(min(frames, len(json_files))):
        json_path = os.path.join(sam_dir, json_files[f_idx])
        
        with open(json_path, 'r') as f:
            sam_data = json.load(f)
        
        # 1. 해당 프레임에서 patient_id와 일치하는 object 찾기
        target_obj = None
        for obj in sam_data.get('objects', []):
            if obj.get('id') == patient_id: # JSON의 id와 patient_id 매칭
                target_obj = obj
                break
        
        if target_obj is None:
            continue # 해당 프레임에 대상 환자가 없으면 스킵
            
        # 2. 마스크 복원 (Fortran order 'F' 사용)
        seg = target_obj['segmentation']
        h, w = seg['size']
        counts = seg['counts']
        
        mask_flat = np.zeros(h * w, dtype=np.uint8)
        current_pos = 0
        val = 0 
        for count in counts:
            mask_flat[current_pos : current_pos + count] = val
            current_pos += count
            val = 1 - val
        
        # SAM/COCO 표준인 세로 방향(F)으로 복원
        binary_mask = mask_flat.reshape((h, w), order='F')

        # 3. 마스크 내부 여부 확인 및 제거
        for k in range(n_kpts):
            x, y = filtered_np[f_idx, k, 0], filtered_np[f_idx, k, 1]
            if np.isnan(x) or np.isnan(y): continue

            ix, iy = int(round(x)), int(round(y))
            
            if 0 <= ix < w and 0 <= iy < h:
                # 🌟 [핵심 변경] 마스크 내부(1)에 있으면 제거하여 보간 대상(NaN)으로 만듦
                if binary_mask[iy, ix] == 1: 
                    filtered_np[f_idx, k, :2] = np.nan

    # 4. 제거된(NaN) 구간 선형 보간
    for k in range(n_kpts):
        series_x = pd.Series(filtered_np[:, k, 0])
        series_y = pd.Series(filtered_np[:, k, 1])
        filtered_np[:, k, 0] = series_x.interpolate(method='linear', limit_direction='both').to_numpy()
        filtered_np[:, k, 1] = series_y.interpolate(method='linear', limit_direction='both').to_numpy()
        
    return filtered_np