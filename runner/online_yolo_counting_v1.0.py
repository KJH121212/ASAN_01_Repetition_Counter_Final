import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# 🌟 새로 추가된 YOLO 라이브러리
from ultralytics import YOLO

# =================================================================
# 1. 경로 설정 및 모듈 임포트
# =================================================================
print("📋 [Step 0] 초기 설정 및 모듈 로드")
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = DATA_DIR / "metadata_v2.1.csv"
YOLO_PATH = DATA_DIR / "checkpoints" / "YOLO_FINETUNING" / "v1.0_step1" / "weights"/ "best.pt"

sys.path.append(str(BASE_DIR))

try:
    from utils.path_list import path_list
    from utils.extract_kpt import normalize_skeleton_array  # (JSON 추출 함수는 빼고 정규화만 가져옵니다)
    from utils.counter_core import UniversalRepetitionCounter
    from utils.parser import parse_common_path
    print("✅ 로컬 모듈 로드 완료.")
except ImportError as e:
    print(f"❌ 로컬 모듈 로드 실패: {e}")
    sys.exit(1)

# 🤖 YOLO 모델 로드 (최초 1회만 로드하여 속도 최적화)
print("🤖 YOLO Pose 모델 로드 중...")
try:
    yolo_model = YOLO(YOLO_PATH)
    print("✅ YOLO 모델 로드 완료.")
except Exception as e:
    print(f"❌ YOLO 모델 로드 실패: {e}")
    sys.exit(1)

# =================================================================
# 2. 데이터 필터링
# =================================================================
print("\n📋 [Step 1] 사용 데이터 처리")

df = pd.read_csv(CSV_PATH)
filtered_df = df[df['common_path'].str.contains('biceps_curl', case=False, na=False)].copy()

parsed_results = filtered_df['common_path'].apply(parse_common_path)
filtered_df['angle'] = parsed_results.apply(lambda x: x[0])  
filtered_df['exercise'] = parsed_results.apply(lambda x: x[1])

print(filtered_df[['common_path', 'angle', 'exercise']].head())
print("✅ 파싱 완료")

# =================================================================
# 3. YOLO 추출 및 카운팅 비디오 렌더링 함수
# =================================================================

def generate_yolo_counting_video(frame_dir, output_path, counter, model, conf_threshold=0.5):
    """
    YOLO 추론, 스켈레톤 정규화, 카운팅, 렌더링을 프레임 단위로 동시에 처리하는 실시간 파이프라인입니다.
    """
    
    frame_files = sorted(list(Path(frame_dir).glob("*.jpg")) + list(Path(frame_dir).glob("*.png")))
    if not frame_files:
        print("❌ 프레임 이미지가 없습니다.")
        return

    print(f"  -> 🚀 실시간 YOLO 추론 및 렌더링 동시 진행 중... (총 {len(frame_files)} 프레임)")
    
    # 비디오 렌더러 초기화
    first_img = cv2.imread(str(frame_files[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

    LINKS_17 = [(5,7), (7,9), (6,8), (8,10), (11,13), (13,15), (12,14), (14,16), (5,6), (11,12), (5,11), (6,12)]
    KPT_17_LEFT, KPT_17_RIGHT = {5, 7, 9, 11, 13, 15}, {6, 8, 10, 12, 14, 16}

    # 💡 [핵심] 단일 루프로 통합되었습니다!
    for f in tqdm(frame_files, desc="🎬 실시간 처리 진행률"):
        # ⏱️ FPS 측정을 위한 시작 시간 기록
        start_time = time.time()
        
        # 0. 프레임 로드
        frame = cv2.imread(str(f))
        if frame is None: continue

        # 1. YOLO 추론 (이미지 배열 자체를 바로 모델에 넘깁니다)
        results = model(frame, verbose=False)
        raw_kpt = np.zeros((17, 3)) 
        
        # 사람이 감지되었을 경우
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            boxes = results[0].boxes.xywh.cpu().numpy()
            areas = boxes[:, 2] * boxes[:, 3]
            main_idx = np.argmax(areas) # 메인 타겟(가장 큰 사람) 추출
            raw_kpt = results[0].keypoints.data[main_idx].cpu().numpy() # (17, 3)

        # 2. 스켈레톤 정규화 (1프레임만 처리하기 위해 차원을 늘렸다 줄입니다)
        # raw_kpt (17, 3) -> expand_dims -> (1, 17, 3) -> 정규화 -> [0]으로 꺼냄 -> (17, 3)
        norm_kpt = normalize_skeleton_array(np.expand_dims(raw_kpt, axis=0))[0]

        # 3. 카운트 엔진 실시간 업데이트
        metrics, events = counter.process_frame(norm_kpt)
        
        # 4. 화면에 스켈레톤 뼈대 및 관절점 그리기
        coords = raw_kpt[:, :2].astype(int)
        scores = raw_kpt[:, 2]
        
        for u, v in LINKS_17:
            if scores[u] > conf_threshold and scores[v] > conf_threshold:
                if coords[u][0] > 0 and coords[v][0] > 0:
                    cv2.line(frame, tuple(coords[u]), tuple(coords[v]), (100, 100, 100), 2, cv2.LINE_AA)
        
        for idx, (x, y) in enumerate(coords):
            if 5 <= idx <= 16 and scores[idx] > conf_threshold and x > 0:
                color = (0, 0, 255) if idx in KPT_17_RIGHT else ((255, 0, 0) if idx in KPT_17_LEFT else (0, 255, 0))
                cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)

        # ==================== UI 그리기 영역 ====================
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        
        # [우측 상단] 카운트 표시
        l_count, r_count = counter.counts.get('left', 0), counter.counts.get('right', 0)
        text_l, text_r = f"Left: {l_count}", f"Right: {r_count}"
        (tw_l, th_l), _ = cv2.getTextSize(text_l, font, scale, thickness)
        (tw_r, th_r), _ = cv2.getTextSize(text_r, font, scale, thickness)
        
        x_l, x_r = w - tw_l - 30, w - tw_r - 30
        pad = 12
        cv2.rectangle(frame, (x_l - pad, 60 - th_l - pad), (x_l + tw_l + pad, 60 + pad), (220, 50, 50), -1) 
        cv2.rectangle(frame, (x_r - pad, 130 - th_r - pad), (x_r + tw_r + pad, 130 + pad), (50, 50, 220), -1) 
        cv2.putText(frame, text_l, (x_l, 60), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, text_r, (x_r, 130), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # [우측 하단] 메트릭(각도/거리) 표시
        l_metric, r_metric = metrics.get('left'), metrics.get('right')
        unit = "deg" if counter.calc_method == 'angle' else "dist"
        
        str_l_metric = f"L {unit}: {l_metric:.1f}" if l_metric is not None else f"L {unit}: N/A"
        str_r_metric = f"R {unit}: {r_metric:.1f}" if r_metric is not None else f"R {unit}: N/A"
        
        scale_m, thick_m = 0.8, 2
        (tw_ml, th_ml), _ = cv2.getTextSize(str_l_metric, font, scale_m, thick_m)
        (tw_mr, th_mr), _ = cv2.getTextSize(str_r_metric, font, scale_m, thick_m)
        
        x_ml, y_ml = w - tw_ml - 30, h - 80
        x_mr, y_mr = w - tw_mr - 30, h - 30
        cv2.rectangle(frame, (x_ml - 10, y_ml - th_ml - 10), (x_ml + tw_ml + 10, y_ml + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_mr - 10, y_mr - th_mr - 10), (x_mr + tw_mr + 10, y_mr + 10), (0, 0, 0), -1)
        cv2.putText(frame, str_l_metric, (x_ml, y_ml), font, scale_m, (255, 200, 200), thick_m, cv2.LINE_AA)
        cv2.putText(frame, str_r_metric, (x_mr, y_mr), font, scale_m, (200, 200, 255), thick_m, cv2.LINE_AA)

        # [좌측 상단] 실시간 FPS 표시 (추론 + 정규화 + 그리기 모두 합친 속도)
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0.0
        fps_text = f"FPS: {fps:.1f}"
        
        (tw_fps, th_fps), _ = cv2.getTextSize(fps_text, font, 1.0, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw_fps + 20, 10 + th_fps + 20), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (20, 10 + th_fps + 10), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # 최종 프레임 비디오 파일에 쓰기
        out.write(frame)

    out.release()
    print(f"  -> ✅ 실시간 처리 비디오 저장 완료: {output_path}")
# =================================================================
# 4. 메인 실행 루프
# =================================================================
print("\n🚀 [Step 3] 개별 데이터 YOLO 추출 및 렌더링 시작...")

TEST_LIMIT = 3 
df_to_process = filtered_df.head(TEST_LIMIT) if TEST_LIMIT is not None else filtered_df

print(f"👉 총 {len(df_to_process)}개의 시퀀스를 처리합니다.")

for idx, row in df_to_process.iterrows():
    common_path = row['common_path']
    camera_angle = row['angle']   
    exercise_name = row['exercise'] 
    
    paths = path_list(common_path, create_dirs=True)
    print(f"\n▶️ 처리 중 [{idx}]: {common_path}")

    try:
        # 비디오 렌더링용 카운터 객체 생성
        counter = UniversalRepetitionCounter(exercise_name=exercise_name, camera_angle=camera_angle)
        
        # 출력 파일 경로 설정
        output_mp4 = paths['test'] / "yolo_counting_video_online_finetuned.mp4"
        
        # YOLO 통합 함수 호출
        generate_yolo_counting_video(
            frame_dir=str(paths['frame']),
            output_path=str(output_mp4),
            counter=counter,
            model=yolo_model # 전역에서 로드한 YOLO 모델 전달
        )
        
        print(f"🎯 [Index {idx}] 최종 결과: 왼팔 {counter.counts['left']}회 / 오른팔 {counter.counts['right']}회")
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ [오류] Index {idx} 처리 중 에러 발생: {e}")

print("\n✨ 모든 시퀀스 처리가 완료되었습니다!")