import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. COCO 17 Keypoints 명칭 정의 (필수)
# ==========================================
KPT_NAMES = {
    0: "Nose", 
    1: "L-Eye", 2: "R-Eye", 
    3: "L-Ear", 4: "R-Ear",
    5: "L-Shoulder", 6: "R-Shoulder", 
    7: "L-Elbow", 8: "R-Elbow",
    9: "L-Wrist", 10: "R-Wrist", 
    11: "L-Hip", 12: "R-Hip",
    13: "L-Knee", 14: "R-Knee", 
    15: "L-Ankle", 16: "R-Ankle"
}

# ==========================================
# 2. 시각화 및 저장 함수
# ==========================================
def plot_and_save_keypoint_analysis(data_array, save_path=None):
    """
    (Frames, 17, 3) 데이터를 시각화합니다.
    - 1열: X좌표, 2열: Y좌표, 3열: 이동 속도
    """
    n_frames = data_array.shape[0]
    
    # 1. 이동 속도 계산 (Frames, 17)
    # 
    diff = np.diff(data_array[:, :, :2], axis=0) 
    velocity = np.linalg.norm(diff, axis=2) 
    velocity = np.vstack([np.zeros((1, 17)), velocity]) 

    # 2. 그래프 설정 (17행 3열)
    fig, axes = plt.subplots(17, 3, figsize=(18, 50), sharex=True, constrained_layout=True)
    frames = np.arange(n_frames)

    for i in range(17):
        # 관절 이름 가져오기
        name = KPT_NAMES.get(i, f"KPT_{i}")
        
        # --- 1열: X좌표 ---
        ax_x = axes[i, 0]
        ax_x.plot(frames, data_array[:, i, 0], color='royalblue', linewidth=1.5)
        ax_x.set_ylabel(f"{name}\n(X)", fontsize=10, rotation=0, labelpad=40)
        ax_x.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax_x.set_title("X-Coordinate", fontsize=14, fontweight='bold')

        # --- 2열: Y좌표 ---
        ax_y = axes[i, 1]
        ax_y.plot(frames, data_array[:, i, 1], color='crimson', linewidth=1.5)
        ax_y.set_ylabel(f"(Y)", fontsize=10, rotation=0, labelpad=20)
        ax_y.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax_y.set_title("Y-Coordinate", fontsize=14, fontweight='bold')

        # --- 3열: 속도(Velocity) ---
        ax_v = axes[i, 2]
        ax_v.plot(frames, velocity[:, i], color='forestgreen', linewidth=1.2)
        ax_v.set_ylabel(f"(Vel)", fontsize=10, rotation=0, labelpad=20)
        ax_v.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax_v.set_title("Velocity", fontsize=14, fontweight='bold')

    # 마지막 행 X축 라벨
    for j in range(3):
        axes[16, j].set_xlabel("Frame Index")

    # 3. 저장 및 출력
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        print(f"✅ 그래프가 저장되었습니다: {save_path}")

    plt.show()

    import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. Body 12 Keypoints 명칭 정의 (5~16번 슬라이싱 결과)
# ==========================================
# (N, 12, 3)으로 변환된 데이터의 인덱스 매핑입니다.
KPT_NAMES_12 = {
    0: "L-Shoulder", 1: "R-Shoulder",
    2: "L-Elbow",    3: "R-Elbow",
    4: "L-Wrist",    5: "R-Wrist",
    6: "L-Hip",      7: "R-Hip",
    8: "L-Knee",     9: "R-Knee",
    10: "L-Ankle",   11: "R-Ankle"
}

# ==========================================
# 2. 12 KPT 시각화 및 저장 함수
# ==========================================
def plot_and_save_12kpt_analysis(data_array, save_path=None):
    """
    (Frames, 12, 3) 데이터를 시각화합니다.
    - 1열: X좌표, 2열: Y좌표, 3열: 이동 속도
    """
    # 입력 데이터의 형태 확인 (N, 12, 3)
    n_frames, n_kpts, _ = data_array.shape
    if n_kpts != 12:
        print(f"⚠️ 경고: 입력 데이터의 키포인트 개수가 {n_kpts}개입니다. (12개 기대)")

    # 1. 이동 속도 계산 (Frames, 12)
    # 프레임 간 유클리드 거리를 계산하여 속도를 도출합니다.
    diff = np.diff(data_array[:, :, :2], axis=0) 
    velocity = np.linalg.norm(diff, axis=2) 
    velocity = np.vstack([np.zeros((1, n_kpts)), velocity]) # 첫 프레임 속도는 0으로 채움

    # 2. 그래프 설정 (12행 3열)
    # 관절 개수에 맞춰 행 수를 12개로 조정합니다.
    fig, axes = plt.subplots(12, 3, figsize=(18, 36), sharex=True, constrained_layout=True)
    frames = np.arange(n_frames)

    for i in range(12):
        # 관절 이름 가져오기
        name = KPT_NAMES_12.get(i, f"KPT_{i}")
        
        # --- 1열: X좌표 (데이터의 첫 번째 차원) ---
        ax_x = axes[i, 0]
        ax_x.plot(frames, data_array[:, i, 0], color='royalblue', linewidth=1.5)
        ax_x.set_ylabel(f"{name}\n(X)", fontsize=10, rotation=0, labelpad=45)
        ax_x.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax_x.set_title("X-Coordinate", fontsize=14, fontweight='bold')

        # --- 2열: Y좌표 (데이터의 두 번째 차원) ---
        ax_y = axes[i, 1]
        ax_y.plot(frames, data_array[:, i, 1], color='crimson', linewidth=1.5)
        ax_y.set_ylabel(f"(Y)", fontsize=10, rotation=0, labelpad=20)
        ax_y.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax_y.set_title("Y-Coordinate", fontsize=14, fontweight='bold')

        # --- 3열: 속도(Velocity) ---
        ax_v = axes[i, 2]
        ax_v.plot(frames, velocity[:, i], color='forestgreen', linewidth=1.2)
        ax_v.set_ylabel(f"(Vel)", fontsize=10, rotation=0, labelpad=20)
        ax_v.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax_v.set_title("Velocity", fontsize=14, fontweight='bold')

    # 마지막 행 X축 라벨 설정
    for j in range(3):
        axes[11, j].set_xlabel("Frame Index")

    # 3. 파일 저장
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        print(f"✅ 12KPT 분석 그래프가 저장되었습니다: {save_path}")

    plt.show()