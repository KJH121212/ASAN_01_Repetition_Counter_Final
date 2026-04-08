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

def plot_and_save_12kpt_optimized(data_array, save_path=None):
    """
    (Frames, 12, 3) 데이터를 고속으로 시각화합니다.
    """
    n_frames, n_kpts, _ = data_array.shape
    frames = np.arange(n_frames)
    
    # 1. NumPy 벡터 연산 최적화 (속도 계산)
    # diff를 한 번만 계산하여 재사용
    coords = data_array[:, :, :2]
    velocity = np.zeros((n_frames, n_kpts))
    velocity[1:] = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=2))

    # 2. Matplotlib 전역 설정 최적화 (루프 내 오버헤드 감소)
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
    
    # Figure 생성 시 constrained_layout 대신 원시적인 조절이 더 빠를 때가 있음
    fig, axes = plt.subplots(12, 3, figsize=(16, 30), sharex=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.15) # 간격 수동 설정이 더 빠름

    titles = ["X-Coordinate", "Y-Coordinate", "Velocity"]
    colors = ['royalblue', 'crimson', 'forestgreen']
    kpt_names = ["(0)L-Shoulder", "(1)R-Shoulder", "(2)L-Elbow", "(3)R-Elbow", "(4)L-Wrist", "(5)R-Wrist", 
                 "(6)L-Hip", "(7)R-Hip", "(8)L-Knee", "(9)R-Knee", "(10)L-Ankle", "(11)R-Ankle"]

    # 3. 루프 최적화
    for i in range(n_kpts):
        # 각 열 데이터를 미리 추출하여 인덱싱 부하 감소
        row_data = [data_array[:, i, 0], data_array[:, i, 1], velocity[:, i]]
        
        for j in range(3):
            ax = axes[i, j]
            # plot 호출 시 label 등을 생략하여 오버헤드 감소
            ax.plot(frames, row_data[j], color=colors[j], linewidth=1.2)
            
            # 첫 번째 행에만 타이틀 부여
            if i == 0:
                ax.set_title(titles[j], fontsize=12, fontweight='bold')
            
            # 첫 번째 열에만 관절 명칭 부여
            if j == 0:
                ax.set_ylabel(kpt_names[i], fontsize=10, fontweight='bold', rotation=0, labelpad=40)

    # 4. 일괄 레이아웃 정리
    for j in range(3):
        axes[-1, j].set_xlabel("Frame Index")

    # 5. 저장 프로세스
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # dpi를 낮추거나 필요한 만큼만 설정 (dpi=80도 분석용으론 충분)
        plt.savefig(save_path, dpi=90, bbox_inches='tight')
        print(f"✅ 저장 완료: {save_path}")

    # 메모리 해제 필수
    plt.close(fig)