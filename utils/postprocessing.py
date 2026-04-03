from pathlib import Path # 파일 경로 처리를 위해 Path 객체를 가져옵니다.
import json # JSON 파일 읽기/쓰기를 위해 json 모듈을 가져옵니다.
from tqdm import tqdm # 진행 상황을 시각적으로 보여주기 위해 tqdm을 가져옵니다.
from typing import Literal, List, Optional, Union # 타입 힌팅을 통해 코드 가독성과 안정성을 높입니다.
import numpy as np # 수학 연산 및 배열 처리를 위해 numpy를 가져옵니다.
from utils.extract_kpt import extract_id_keypoints, normalize_skeleton_array
import pandas as pd # 선형 보간을 위해 pandas를 가져옵니다.
from scipy import stats # 최빈값(mode)을 빠르고 쉽게 계산하기 위해 scipy의 stats 모듈을 불러옵니다.

# ==========================================
# 함수 1: 특정 인물의 데이터만 골라내어 별도 폴더에 저장
# ==========================================
def apply_axis_selective_iqr_filter(
    data_np: np.ndarray, 
    target_kpts: Optional[List[int]] = None,
    max_pixel_speed: float = 50.0, 
    iqr_multiplier: float = 3.0, 
    use_iqr: bool = True,
    axis: Literal['x', 'y', 'both'] = 'both'
) -> np.ndarray:
    """
    특정 관절(target_kpts)과 특정 축(axis)을 선택하여 IQR 기반 이상치 보정을 수행합니다.
    
    Args:
        data_np: (Frames, Kpts, 3) 형태의 넘파이 배열.
        target_kpts: 필터를 적용할 관절 인덱스 리스트 (예: [10, 11] - 양발). 
                     None이면 모든 관절에 적용합니다.
        max_pixel_speed: 물리적 한계 속도.
        iqr_multiplier: IQR 가중치. (1.5: 엄격, 3.0: 보통)
        axis: 필터링할 축 설정 ('x', 'y', 'both').
    """
    filtered_np = data_np.copy()
    frames, n_kpts, _ = filtered_np.shape
    
    # 대상 관절이 지정되지 않았다면 전체 관절을 대상으로 설정합니다.
    if target_kpts is None:
        target_kpts = list(range(n_kpts))
        
    for k in target_kpts:
        # 1. 축별 속도 데이터 확보
        v_x = np.zeros(frames)
        v_y = np.zeros(frames)
        # 각 프레임 간의 좌표 차이(절대값)를 계산하여 속도로 간주합니다.
        v_x[1:] = np.abs(filtered_np[1:, k, 0] - filtered_np[:-1, k, 0])
        v_y[1:] = np.abs(filtered_np[1:, k, 1] - filtered_np[:-1, k, 1])

        # 2. IQR 임계값 계산
        thresholds = {'x': max_pixel_speed, 'y': max_pixel_speed}
        if use_iqr:
            for ax_name, v_data in zip(['x', 'y'], [v_x, v_y]):
                valid_v = v_data[(v_data > 0) & (v_data <= max_pixel_speed)]
                if len(valid_v) > 5:
                    q1, q3 = np.percentile(valid_v, [25, 75])
                    iqr = q3 - q1
                    stat_threshold = q3 + (iqr_multiplier * iqr)
                    thresholds[ax_name] = min(max_pixel_speed, stat_threshold)

        # 3. 이상치 탐지 및 NaN 처리
        last_valid_x = None
        last_valid_y = None
        
        for f in range(frames):
            curr_x, curr_y = filtered_np[f, k, 0], filtered_np[f, k, 1]
            score = filtered_np[f, k, 2]
            
            # 신뢰도가 낮거나 좌표가 (0,0)인 데이터는 기본적으로 제거 대상
            is_invalid = (score <= 0 or (curr_x == 0 and curr_y == 0))
            
            # X축 체크
            if is_invalid or (axis in ['x', 'both'] and last_valid_x is not None and 
                             np.abs(curr_x - last_valid_x) > thresholds['x']):
                filtered_np[f, k, 0] = np.nan
            else:
                last_valid_x = curr_x
            
            # Y축 체크
            if is_invalid or (axis in ['y', 'both'] and last_valid_y is not None and 
                             np.abs(curr_y - last_valid_y) > thresholds['y']):
                filtered_np[f, k, 1] = np.nan
            else:
                last_valid_y = curr_y

        # 4. Pandas를 이용한 선형 보간 수행
        if axis in ['x', 'both']:
            s_x = pd.Series(filtered_np[:, k, 0])
            filtered_np[:, k, 0] = s_x.interpolate(method='linear', limit_direction='both').to_numpy()
            
        if axis in ['y', 'both']:
            s_y = pd.Series(filtered_np[:, k, 1])
            filtered_np[:, k, 1] = s_y.interpolate(method='linear', limit_direction='both').to_numpy()

    return filtered_np

# ==========================================
# class 1 : oulier 발생시, 이전 정상 위치를 유지
# ==========================================
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

# ==========================================
# class 2 : oulier 발생시, 예측값을 따라 정상값 변경
# ==========================================
class VelocityPredictiveKalman:
    def __init__(self, threshold=30.0, q_std=0.01, r_std=0.5):
        self.threshold = threshold 
        self.dt = 1.0
        self.x = None  # [x, y, vx, vy]
        self.P = np.eye(4)
        
        # 상태 전이 행렬 (등속 모델)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * (q_std**2)
        self.R = np.eye(2) * (r_std**2)

    def process_step(self, z):
        """
        한 프레임에 대한 예측 및 업데이트 로직 (내부 계산용)
        """
        z_col = np.array(z).reshape(2, 1)
        
        if self.x is None:
            self.x = np.zeros((4, 1))
            self.x[:2] = z_col
            return z_col.flatten(), z_col.flatten() # (실제 결과, 모델 예측값)

        # 1. 예측 (Predict)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        pos_pred = (self.H @ x_pred).flatten()
        
        # 2. 잔차 및 이상치 판별
        dist = np.linalg.norm(z.flatten() - pos_pred)
        
        if dist > self.threshold:
            # 🚨 Outlier 발생: 예측값 채택
            self.x = x_pred
            self.P = P_pred
            return pos_pred, pos_pred # 결과값으로 예측치를 사용
        else:
            # 🟢 정상: 칼만 업데이트
            y = z_col - (self.H @ x_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            self.x = x_pred + K @ y
            self.P = (np.eye(4) - K @ self.H) @ P_pred
            return (self.H @ self.x).flatten(), pos_pred

# ==========================================
# 함수 2: IQR 방식을 이용해서 outlier를 찾고 빈자리를 선형 보간으로 체워주는 함수
# ==========================================
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

# ==========================================
# 함수 3: 칼만 필터 기반 outlier 변경 (이전 값 복사)
# ==========================================
def apply_axis_selective_kalman(data_np, threshold=0.1, q_std=0.05, r_std=0.01, target_kpts=None, axis='both'):
    """
    특정 축(X, Y, 혹은 둘 다)을 선택하여 칼만 필터를 적용합니다.
    
    Args:
        data_np: 원본 넘파이 배열 (N, 12, 3)
        threshold, q_std, r_std: 칼만 필터 파라미터
        target_kpts: 필터를 적용할 관절 인덱스 리스트
        axis: 필터를 적용할 축 설정 ('x', 'y', 'both')
    """
    frames, n_kpts, _ = data_np.shape # 데이터의 차원 정보를 가져옵니다.
    filtered_np = data_np.copy() # 원본 데이터 보존을 위해 복사본을 생성합니다.
    
    # 각 관절마다 독립적인 필터 인스턴스를 생성합니다.
    filters = [PredictiveKalmanFilter(threshold=threshold, q_std=q_std, r_std=r_std) for _ in range(n_kpts)]
    
    # 대상 관절이 지정되지 않았다면 모든 관절에 대해 수행합니다.
    if target_kpts is None:
        target_kpts = list(range(n_kpts))
        
    for f in range(frames): # 모든 프레임을 순회합니다.
        for k in target_kpts: # 지정된 관절들만 순회합니다.
            z = data_np[f, k, :2].copy() # 현재 프레임의 x, y 좌표를 가져옵니다.
            score = data_np[f, k, 2] # 현재 프레임의 신뢰도 점수를 가져옵니다.
            
            # 관절이 유효하게 인식된 경우
            if score > 0 and not np.all(z == 0):
                # 칼만 필터 프로세스 실행 (내부적으로 x, y 모두 처리)
                kf_result = filters[k].process(z) # 필터링된 [x, y] 결과를 얻습니다.
                
                # --- [축 선택 로직] ---
                if axis == 'x':
                    filtered_np[f, k, 0] = kf_result[0] # X축만 필터링된 값을 적용합니다.
                    # Y축은 원본(data_np[f, k, 1])을 유지합니다.
                elif axis == 'y':
                    filtered_np[f, k, 1] = kf_result[1] # Y축만 필터링된 값을 적용합니다.
                    # X축은 원본(data_np[f, k, 0])을 유지합니다.
                else: # 'both'인 경우
                    filtered_np[f, k, :2] = kf_result # X, Y 모두 필터링된 값을 적용합니다.
            
            # 관절을 놓친 경우 (결측치 처리)
            else:
                if filters[k].last_valid_z is not None:
                    # 필터 내부 상태 업데이트 (속도 0으로 고정)
                    filters[k].x[:2] = filters[k].last_valid_z.reshape(2, 1)
                    filters[k].x[2:] = 0 
                    
                    # 결측치 구간에서도 선택된 축에 따라 마지막 유효값을 채워넣습니다.
                    if axis == 'x':
                        filtered_np[f, k, 0] = filters[k].last_valid_z[0]
                    elif axis == 'y':
                        filtered_np[f, k, 1] = filters[k].last_valid_z[1]
                    else:
                        filtered_np[f, k, :2] = filters[k].last_valid_z
                        
    return filtered_np # 최종 결과 배열을 반환합니다.

# ==========================================
# 함수 3: 칼만 필터 기반 outlier 변경 (예측값 사용)
# ==========================================
def apply_axis_velocity_kalman(data_np, threshold=25.0, q_std=0.01, r_std=0.5, target_kpts=None, axis='both'):
    """
    이상치 발생 시 예측값을 사용하며, X/Y 축별로 선택적 적용이 가능한 함수입니다.
    """
    frames, n_kpts, _ = data_np.shape
    filtered_np = data_np.copy()
    
    filters = [VelocityPredictiveKalman(threshold=threshold, q_std=q_std, r_std=r_std) for _ in range(n_kpts)]
    
    if target_kpts is None:
        target_kpts = list(range(n_kpts))

    for f in range(frames):
        for k in target_kpts:
            z_orig = data_np[f, k, :2].copy()
            score = data_np[f, k, 2]
            
            if score > 0 and not np.all(z_orig == 0):
                # 필터링 결과와 모델의 예측값을 가져옵니다.
                kf_res, pred_res = filters[k].process_step(z_orig)
                
                # --- [축 선택 적용 로직] ---
                if axis == 'x':
                    filtered_np[f, k, 0] = kf_res[0] # X만 필터값(혹은 예측값) 적용
                elif axis == 'y':
                    filtered_np[f, k, 1] = kf_res[1] # Y만 필터값(혹은 예측값) 적용
                else: # 'both'
                    filtered_np[f, k, :2] = kf_res
            else:
                # 데이터가 없을 때(결측치)는 무조건 예측 경로를 따라갑니다.
                if filters[k].x is not None:
                    x_next = filters[k].F @ filters[k].x
                    filters[k].x = x_next
                    filters[k].P = filters[k].F @ filters[k].P @ filters[k].F.T + filters[k].Q
                    pos_only = (filters[k].H @ x_next).flatten()
                    
                    if axis == 'x': filtered_np[f, k, 0] = pos_only[0]
                    elif axis == 'y': filtered_np[f, k, 1] = pos_only[1]
                    else: filtered_np[f, k, :2] = pos_only
                    
    return filtered_np

# ==========================================
# 함수 4: 칼만 필터 기반 smoothing 적용(RTS:Rauch Tung Strievel)
# ==========================================
def apply_kalman_smoothing(data_np, q_std=0.01, r_std=0.1, target_kpts=None, axis='both'):
    """
    RTS(Rauch-Tung-Striebel) Smoother를 적용하여 전체 경로를 부드럽게 만듭니다.
    일반 칼만 필터보다 지연(Lag)이 적고 곡선이 훨씬 매끄럽습니다.
    
    Args:
        data_np: (Frames, Kpts, 3) 넘파이 배열
        q_std: 과정 노이즈 (낮을수록 경로가 직선에 가까워짐)
        r_std: 측정 노이즈 (높을수록 원본 값을 덜 믿고 부드럽게 만듦)
        target_kpts: 적용할 관절 리스트
    """
    frames, n_kpts, _ = data_np.shape
    filtered_np = data_np.copy()
    
    if target_kpts is None:
        target_kpts = list(range(n_kpts))

    dt = 1.0
    # 등속 모델 설정
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    Q = np.eye(4) * (q_std**2)
    R = np.eye(2) * (r_std**2)

    for k in target_kpts:
        # 1. Forward Pass (일반 칼만 필터링)
        x_history = np.zeros((frames, 4, 1))
        P_history = np.zeros((frames, 4, 4))
        x_preds = np.zeros((frames, 4, 1))
        P_preds = np.zeros((frames, 4, 4))
        
        curr_x = np.zeros((4, 1))
        curr_x[:2] = data_np[0, k, :2].reshape(2, 1)
        curr_P = np.eye(4)

        for f in range(frames):
            # Predict
            x_pred = F @ curr_x
            P_pred = F @ curr_P @ F.T + Q
            
            x_preds[f] = x_pred
            P_preds[f] = P_pred
            
            # Update
            z = data_np[f, k, :2].reshape(2, 1)
            score = data_np[f, k, 2]
            
            if score > 0 and not np.all(z == 0):
                y = z - (H @ x_pred)
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                curr_x = x_pred + K @ y
                curr_P = (np.eye(4) - K @ H) @ P_pred
            else:
                curr_x, curr_P = x_pred, P_pred
                
            x_history[f] = curr_x
            P_history[f] = curr_P

        # 2. Backward Pass (RTS Smoothing)
        # 마지막 프레임부터 거꾸로 계산하여 미래 정보를 현재에 반영합니다.
        smoothed_x = x_history.copy()
        for f in range(frames - 2, -1, -1):
            # Smoothing Gain
            C = P_history[f] @ F.T @ np.linalg.inv(P_preds[f+1])
            smoothed_x[f] = x_history[f] + C @ (smoothed_x[f+1] - x_preds[f+1])

        # 3. 결과 반영
        for f in range(frames):
            res = (H @ smoothed_x[f]).flatten()
            if axis == 'x': filtered_np[f, k, 0] = res[0]
            elif axis == 'y': filtered_np[f, k, 1] = res[1]
            else: filtered_np[f, k, :2] = res

    return filtered_np

# ==========================================
# 함수 5: 최빈값, 중앙값 등의 방식으로 kpt 고정하는 함수
# ==========================================
def fix_keypoints_to_stat(data_np, target_kpts, axis='both', method='binned_mode', bin_size=10.0, min_score=0.05, frame_idx=None): # 특정 프레임 지정 및 다양한 통계 방식을 지원하는 키포인트 고정 함수를 정의합니다.
    """
    주어진 데이터에서 특정 키포인트의 좌표를 지정한 통계 방식 또는 특정 프레임의 좌표값으로 전체 프레임에 걸쳐 고정합니다.
    """
    filtered_np = data_np.copy() # 원본 배열이 변형되지 않도록 깊은 복사(copy)를 생성하여 안전하게 작업합니다.

    if isinstance(target_kpts, int): # 사용자가 타겟 키포인트를 단일 숫자(int)로 입력했는지 유연하게 검사합니다.
        target_kpts = [target_kpts]  # 단일 숫자라도 반복문(for)에서 에러가 나지 않도록 리스트 형태로 감싸줍니다.

    for k in target_kpts: # 고정하고자 하는 모든 타겟 키포인트에 대해 순차적으로 반복 작업을 수행합니다.
        target_val = np.zeros(2) # 계산된 최종 대푯값(x, y)을 임시로 저장해 둘 배열을 0으로 초기화하여 준비합니다.

        if method == 'specific_frame': # 사용자가 선택한 방식이 '특정 프레임 고정(specific_frame)'인지 확인합니다.
            if frame_idx is None or frame_idx < 0 or frame_idx >= data_np.shape[0]: # 프레임 인덱스가 유효한 범위를 벗어나는지 방어적으로 검사합니다.
                raise ValueError("올바른 frame_idx를 입력해야 합니다.") # 잘못된 인덱스일 경우 에러를 발생시켜 잘못된 참조를 미연에 방지합니다.
            
            target_val = data_np[frame_idx, k, :2] # 전체 데이터를 순회할 필요 없이, 지정한 단일 프레임의 특정 키포인트 x, y 좌표를 즉시 대푯값으로 가져옵니다.

        else: # 특정 프레임 지정 방식이 아닌, 기존의 전체 프레임 기반 통계 방식들을 처리하는 구간입니다.
            valid_idx = data_np[:, k, 2] > min_score # 신뢰도(score)가 기준치보다 높은, 즉 의미 있는 데이터가 있는 프레임의 위치(인덱스)를 찾습니다.
            
            if not np.any(valid_idx): # 만약 유효한 데이터가 단 한 프레임도 존재하지 않는지 검사합니다.
                continue # 유효한 데이터가 없다면 통계를 낼 수 없으므로, 아무 작업 없이 다음 키포인트로 건너뜁니다.

            valid_data = data_np[valid_idx, k, :2] # 0이나 결측치로 인해 대푯값이 왜곡되는 것을 막기 위해, 유효한 프레임의 x와 y 좌표만 (N, 2) 형태로 따로 추출합니다.

            if method == 'mean': # 사용자가 선택한 통계 방식이 '평균값(mean)'인지 확인합니다.
                target_val = np.mean(valid_data, axis=0) # 유효한 x, y 좌표들의 산술 평균을 일괄적으로 계산하여 대푯값으로 삼습니다.
                
            elif method == 'median': # 사용자가 선택한 통계 방식이 '중앙값(median)'인지 확인합니다.
                target_val = np.median(valid_data, axis=0) # 유효한 x, y 좌표들을 크기순으로 나열했을 때 정중앙에 있는 값을 계산합니다.
                
            elif method == 'binned_mode': # 새롭게 개선된 '2D 묶음 기반 최빈 좌표' 추출 방식입니다.
                quantized_data = np.floor(valid_data / bin_size).astype(int) # x, y 좌표를 bin_size로 나누고 내림하여 2D 격자(Grid) 인덱스 정수로 변환합니다.
                
                unique_bins, counts = np.unique(quantized_data, axis=0, return_counts=True) # 격자화된 2D 배열에서 중복을 제거하고 각 격자의 빈도수를 셉니다.
                
                max_count_idx = np.argmax(counts) # 빈도수(counts) 배열에서 가장 높은 숫자가 있는 위치(인덱스)를 찾아냅니다.
                
                best_bin = unique_bins[max_count_idx] # 가장 많이 등장한 1등 격자의 [x, y] 인덱스 값을 가져옵니다.
                
                in_best_bin_mask = np.all(quantized_data == best_bin, axis=1) # 전체 quantized_data 중 1등 격자와 [x, y]가 정확히 일치하는 행들만 True로 마스킹합니다.
                
                best_real_data = valid_data[in_best_bin_mask] # 마스크를 이용해 1등 격자에 속했던 '원본 실수 좌표'들만 걸러냅니다.
                
                target_val = np.median(best_real_data, axis=0) # 걸러낸 진짜 좌표들의 중앙값을 구해 미세한 노이즈를 한 번 더 제거한 최종 단일 좌표를 확정합니다.

            else: # 사용자가 지원하지 않는 오타나 잘못된 방식을 입력했는지 방어적으로 확인합니다.
                raise ValueError("method는 'mean', 'median', 'binned_mode', 'specific_frame' 중 하나여야 합니다.") # 에러를 발생시켜 프로그램 중단을 알립니다.

        # --- [축 선택 적용 로직] ---
        if axis in ['x', 'both']: # 사용자가 x축을 고정하고 싶어 하거나(x), 두 축 모두 고정하고 싶어 하는지(both) 확인합니다.
            filtered_np[:, k, 0] = target_val[0] # 전체 프레임(모든 시간대)에 대해 해당 키포인트의 x 좌표를 위에서 구한 대푯값으로 일괄 덮어씌웁니다.
            
        if axis in ['y', 'both']: # 사용자가 y축을 고정하고 싶어 하거나(y), 두 축 모두 고정하고 싶어 하는지(both) 확인합니다.
            filtered_np[:, k, 1] = target_val[1] # 전체 프레임(모든 시간대)에 대해 해당 키포인트의 y 좌표를 위에서 구한 대푯값으로 일괄 덮어씌웁니다.

    return filtered_np # 지정된 키포인트의 좌표가 모든 조건에 맞게 성공적으로 고정된 최종 데이터 배열을 반환합니다.

def apply_segment_interpolation(
    data_np: np.ndarray, 
    start_frame: int, 
    end_frame: int, 
    target_kpts: Union[int, List[int]], 
    axis: Literal['x', 'y', 'both'] = 'both'
) -> np.ndarray:
    """
    지정된 프레임 구간(start_frame ~ end_frame)의 데이터를 삭제하고 선형 보간을 수행합니다.
    """
    filtered_np = data_np.copy() # 원본 데이터 보존을 위해 복사본을 생성합니다.
    
    # 입력받은 target_kpts가 단일 정수일 경우 리스트로 변환하여 반복문을 지원합니다.
    if isinstance(target_kpts, int):
        target_kpts = [target_kpts] # 단일 인덱스를 리스트화합니다.

    for k in target_kpts: # 지정된 모든 관절에 대해 루프를 돕니다.
        # 1. 보간할 구간을 NaN(Not a Number)으로 설정하여 비워둡니다.
        # end_frame + 1을 통해 b 프레임까지 포함하도록 슬라이싱합니다.
        if axis in ['x', 'both']:
            filtered_np[start_frame : end_frame + 1, k, 0] = np.nan # X축 구간 삭제
        if axis in ['y', 'both']:
            filtered_np[start_frame : end_frame + 1, k, 1] = np.nan # Y축 구간 삭제

        # 2. Pandas Series의 interpolate 기능을 활용해 빈 구간을 직선으로 채웁니다.
        if axis in ['x', 'both']:
            s_x = pd.Series(filtered_np[:, k, 0]) # X좌표를 시리즈로 변환합니다.
            filtered_np[:, k, 0] = s_x.interpolate(method='linear', limit_direction='both').to_numpy() # 선형 보간 후 반영합니다.
            
        if axis in ['y', 'both']:
            s_y = pd.Series(filtered_np[:, k, 1]) # Y좌표를 시리즈로 변환합니다.
            filtered_np[:, k, 1] = s_y.interpolate(method='linear', limit_direction='both').to_numpy() # 선형 보간 후 반영합니다.

    return filtered_np # 구간 보간이 완료된 배열을 반환합니다.