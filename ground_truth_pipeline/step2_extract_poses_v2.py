import cv2, json, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances


# numpy → JSON 직렬화 변환 함수
def to_py(obj):
    """numpy 객체를 JSON 직렬화 가능한 Python 객체로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): 
        return obj.tolist()                       # numpy array → list
    if isinstance(obj, (_np.floating,)): 
        return float(obj)                         # numpy float → float
    if isinstance(obj, (_np.integer,)):  
        return int(obj)                           # numpy int → int
    if isinstance(obj, dict):  
        return {k: to_py(v) for k, v in obj.items()}  # dict 내부 재귀 처리
    if isinstance(obj, (list, tuple)): 
        return [to_py(v) for v in obj]            # list/tuple 내부 재귀 변환
    return obj                                    # 기본 타입 그대로 반환


# Keypoints 추출 (Batch 버전)
def extract_keypoints(frame_dir: str, json_dir: str,
                      det_cfg: str, det_ckpt: str,
                      pose_cfg: str, pose_ckpt: str,
                      device: str = "cuda:0",
                      batch_size: int = 8) -> int:
    """
    주어진 프레임 디렉토리에서 사람 감지 + 포즈 추정 후
    각 프레임별 JSON 파일로 keypoints를 저장합니다.

    Args:
        frame_dir (str): 프레임 이미지(.jpg) 폴더 경로
        json_dir (str): JSON 결과를 저장할 폴더 경로
        det_cfg (str): Detector 설정 파일 경로 (mmdet config)
        det_ckpt (str): Detector checkpoint 파일 경로
        pose_cfg (str): Pose estimator 설정 파일 경로 (mmpose config)
        pose_ckpt (str): Pose estimator checkpoint 파일 경로
        device (str): 실행 장치 ("cuda:0" or "cpu")
        batch_size (int): Batch 단위로 처리할 프레임 수

    Returns:
        int: 생성된 JSON 파일 개수
    """

    frame_dir, json_dir = Path(frame_dir), Path(json_dir)

    # JSON 결과 폴더 초기화
    if json_dir.exists():                         # 기존 폴더가 있으면 삭제
        shutil.rmtree(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)   # 새 폴더 생성

    # Detector (사람 검출기)와 Pose Estimator 초기화
    detector = init_detector(det_cfg, det_ckpt, device=device)   # 객체 탐지 모델 로드
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)            # mmpose 호환 파이프라인 적용

    pose_estimator = init_pose_estimator(                        # 포즈 추정 모델 초기화
        pose_cfg, pose_ckpt, device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))  # heatmap 비활성화
    )

    # 프레임 목록 수집
    frames = sorted(list(frame_dir.rglob("*.jpg")))      # jpg 프레임 전체 정렬
    saved = 0                                    # 저장된 JSON 수 카운트

# Batch 단위로 프레임 처리
    for start in tqdm(range(0, len(frames), batch_size), desc="Sapiens", unit="batch"): # 전체 프레임을 배치 사이즈만큼 나누어 루프를 돕니다.
        batch_files = frames[start:start + batch_size]           # 현재 처리할 배치 단위의 파일 경로 목록을 슬라이싱합니다.
        batch_imgs_bgr = [cv2.imread(str(f)) for f in batch_files]  # 목록에 있는 각 경로의 이미지를 BGR 포맷으로 읽어옵니다.
        batch_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_imgs_bgr if img is not None]  # 읽어온 BGR 이미지를 모델이 사용하는 RGB 포맷으로 변환합니다.

        if not batch_imgs: # 만약 이미지를 읽지 못해 리스트가 비어있다면
            continue                                             # 이번 배치를 건너뛰고 다음 배치로 넘어갑니다.

        try: # 예측 중 발생할 수 있는 에러에 대비하여 try-except 블록을 사용합니다.
            # 사람 감지 (Batch)
            dets = inference_detector(detector, batch_imgs)     # 감지 모델을 통해 배치 이미지 내의 사람 바운딩 박스를 예측합니다.

            # 각 프레임별 포즈 추정
            for idx_in_batch, (fpath, img_rgb, det) in enumerate(zip(batch_files, batch_imgs, dets)): # 파일경로, 이미지, 감지결과를 하나씩 묶어 순회합니다.
                idx_frame = start + idx_in_batch                 # 전체 데이터셋 기준에서의 현재 프레임 인덱스를 계산합니다.
                pred = det.pred_instances.cpu().numpy()          # GPU에 있는 예측 결과를 CPU로 내리고 numpy 배열로 변환합니다.

                # 사람(label==0)만 추출 + confidence 0.5 이상 필터링
                keep = (pred.labels == 0) & (pred.scores > 0.5) # 감지된 객체 중 사람이면서 신뢰도가 0.5가 넘는 조건(마스크)을 생성합니다.
                bbs = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)[keep] # 조건에 맞는 바운딩 박스와 신뢰도 점수를 결합하여 필터링합니다.

                if len(bbs) == 0: # 프레임 내에 조건을 만족하는 사람이 단 한 명도 없다면
                    continue                                     # 포즈 추정을 생략하고 다음 프레임으로 건너뜁니다.

                bbs = bbs[nms(bbs, 0.5), :4]                     # Non-Maximum Suppression을 수행하여 중복된 바운딩 박스를 제거하고 좌표만 남깁니다.

                # 포즈 추정 (각 사람 bounding box별 keypoints 예측)
                pose_results = inference_topdown(pose_estimator, img_rgb, bbs)  # 잘려진 사람 영역(bbs)을 바탕으로 관절 키포인트를 추정합니다.
                data_sample = merge_data_samples(pose_results)   # 여러 명의 사람에 대한 포즈 추정 결과를 하나의 데이터 구조로 통합합니다.
                inst = data_sample.get("pred_instances", None) # 통합된 결과에서 예측된 인스턴스(객체) 정보만 추출합니다.
                if inst is None: # 추출된 인스턴스 정보가 없다면
                    continue # 저장을 생략하고 다음 프레임으로 건너뜁니다.

                inst_list = split_instances(inst)                # 통합되어 있던 인스턴스 정보를 다시 사람(객체)별로 분리하여 리스트로 만듭니다.

                # ------------------------------------------------
                # 8️⃣ JSON 파일로 저장 (하위 폴더 구조 유지)
                # ------------------------------------------------
                payload = dict( # JSON 파일에 담을 데이터 페이로드(내용물) 딕셔너리를 생성합니다.
                    frame_index=idx_frame,                       # 현재 프레임의 전체 인덱스 번호를 기록합니다.
                    meta_info=pose_estimator.dataset_meta,       # 포즈 모델의 스켈레톤 연결 구조 등 메타 정보를 기록합니다.
                    instance_info=inst_list                      # 실제로 추출된 사람별 키포인트 좌표 및 신뢰도 정보를 기록합니다.
                )

                # 💡 핵심 수정 부분: 원본 이미지의 하위 폴더 구조를 파악하여 저장 경로를 동적으로 생성합니다.
                relative_path = fpath.relative_to(frame_dir)
                json_path = json_dir / relative_path.with_suffix('.json') 
                json_path.parent.mkdir(parents=True, exist_ok=True) 

                with open(json_path, "w", encoding="utf-8") as f: # 계산된 경로에 쓰기 모드('w')로 JSON 파일을 엽니다.
                    json.dump(to_py(payload), f, ensure_ascii=False, indent=2) # numpy 데이터 등을 파이썬 기본 타입으로 변환 후 예쁘게 들여쓰기하여 저장합니다.
                saved += 1                                       # 파일이 성공적으로 저장되었으므로 카운트를 1 증가시킵니다.

        except Exception as e: # 배치 처리 중 에러가 발생한 경우를 잡아냅니다.
            print(f"[ERROR] batch {start} → {e}")                # 진행 상황을 방해하지 않도록 에러가 발생한 배치 번호와 내용을 콘솔에 출력합니다.

    # --------------------------------------------------------
    # 9️⃣ 총 저장된 JSON 개수 반환
    # --------------------------------------------------------
    return saved # 모든 배치 처리가 완료된 후 최종적으로 저장된 JSON 파일의 총 개수를 반환합니다.