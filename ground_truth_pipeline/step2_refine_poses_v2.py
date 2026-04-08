import cv2, json  # OpenCV와 JSON 처리 라이브러리
import numpy as np  # 수치 연산용 numpy
from pathlib import Path  # 경로 처리를 위한 Path
from tqdm import tqdm  # 진행 상황 표시용 tqdm
from mmpose.apis import inference_topdown  # 포즈 추정 실행 함수
from mmpose.structures import merge_data_samples, split_instances  # 추론 결과 처리 함수

def to_py(obj):
    """넘파이 객체를 JSON 직렬화 가능한 타입으로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

def reextract_missing_keypoints(
    file_name: str,              # 비디오 파일명
    frame_dir: str,              # 프레임 디렉토리
    json_dir: str,               # JSON 저장 디렉토리
    n_extracted_frames: int,     # 총 추출된 프레임 수
    pose_estimator,              # 초기화된 Sapiens 포즈 추정 모델
) -> int:
    """
    하위 폴더 구조(01, 02...)를 지원하며, 누락된 프레임만 Sapiens로 재추출합니다.
    (bbox는 인접 JSON에서 재활용)
    """
    frame_dir, json_dir = Path(frame_dir), Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # 💡 1. 전체 파일 경로 매핑 (Dictionary 구조 활용)
    # 하위 폴더 위치에 상관없이 파일명(stem)으로 실제 경로를 즉시 찾을 수 있게 만듭니다.
    # ------------------------------------------------
    # 예: {'000000': Path('.../01/000000.jpg'), '010000': Path('.../02/010000.jpg')}
    frame_paths = {p.stem: p for p in frame_dir.rglob("*.jpg")} # 모든 jpg의 하위 경로를 수집합니다.
    json_paths = {p.stem: p for p in json_dir.rglob("*.json")}  # 모든 json의 하위 경로를 수집합니다.

    # 누락된 프레임 번호 계산 (jpg는 존재하지만 json은 없는 프레임 추출)
    missing = sorted(set(frame_paths.keys()) - set(json_paths.keys()))

    if not missing:
        print(f"[INFO] {file_name}: 누락된 프레임 없음")
        return len(json_paths)  # 최종 JSON 개수 반환

    for fidx_str in tqdm(missing, desc=f"{file_name} (re-infer)", unit="frame"):
        fidx = int(fidx_str)
        fpath = frame_paths[fidx_str] # 미리 만들어둔 딕셔너리에서 원본 이미지의 정확한 경로를 가져옵니다.

        # 💡 원본 이미지의 하위 폴더 구조를 본따서 JSON 저장 경로를 동적으로 생성합니다.
        relative_path = fpath.relative_to(frame_dir) 
        jpath = json_dir / relative_path.with_suffix('.json')

        if not fpath.exists() or jpath.exists():
            continue

        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # -------------------------------
        # 💡 2. bbox 재활용 (좌우 인접 JSON 탐색)
        # 이제 딕셔너리(json_paths)를 활용해 인접 파일이 어느 폴더에 있든 쉽게 찾습니다.
        # -------------------------------
        neighbor = None
        off = 1
        while True:
            left_str = f"{fidx-off:06d}" # 왼쪽(과거) 이웃 프레임 번호
            right_str = f"{fidx+off:06d}" # 오른쪽(미래) 이웃 프레임 번호

            if left_str in json_paths: # 딕셔너리 안에 해당 번호의 JSON 경로가 존재한다면
                neighbor = json_paths[left_str] # 해당 경로를 이웃으로 설정합니다.
                break
            if right_str in json_paths: # 딕셔너리 안에 오른쪽 이웃 경로가 존재한다면
                neighbor = json_paths[right_str] # 해당 경로를 이웃으로 설정합니다.
                break
            
            # 탐색 범위가 전체 프레임을 벗어나면 종료합니다.
            if (fidx-off) < 0 and (fidx+off) >= n_extracted_frames:
                break
            off += 1

        if neighbor is None:
            continue

        with open(neighbor, "r", encoding="utf-8") as f: # 찾은 이웃 JSON 파일을 엽니다.
            nb = json.load(f)
        if not nb.get("instance_info"):
            continue

        bbox = np.array(nb["instance_info"][0]["bbox"], dtype=np.float32).reshape(1,4)

        # -------------------------------
        # Sapiens 포즈 재추론
        # -------------------------------
        results = inference_topdown(pose_estimator, img_rgb, bbox)
        data_sample = merge_data_samples(results)
        inst = data_sample.get("pred_instances", None)
        if inst is None:
            continue
        inst_list = split_instances(inst)

        # -------------------------------
        # JSON 저장
        # -------------------------------
        payload = dict(
            frame_index=fidx,
            video_name=file_name,
            meta_info=pose_estimator.dataset_meta,
            instance_info=inst_list,
            source="reextract"
        )
        
        jpath.parent.mkdir(parents=True, exist_ok=True) # 하위 폴더(01, 02 등)가 없으면 생성합니다.
        
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(to_py(payload), f, ensure_ascii=False, indent=2)
            
        # 💡 중요: 방금 새로 생성한 JSON도 다른 누락 프레임의 이웃이 될 수 있으므로 딕셔너리에 추가해 줍니다.
        json_paths[fidx_str] = jpath 

    # 최종 JSON 개수 다시 세서 반환 (rglob 활용)
    final_json_count = len(list(json_dir.rglob("*.json"))) # 하위 폴더 전체의 json 개수를 셉니다.
    print(f"[INFO] {file_name}: 최종 JSON 개수 {final_json_count}")
    return final_json_count

