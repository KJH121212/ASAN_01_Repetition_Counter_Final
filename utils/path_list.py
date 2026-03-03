from pathlib import Path # 경로 처리를 위한 Path 객체 임포트

def path_list(common_path: str, create_dirs: bool = False):
    """
    재활 의학 데이터 처리를 위한 주요 경로들을 정의하고 딕셔너리로 반환합니다.
    
    Args:
        common_path (str): 하위 데이터 그룹의 공통 경로 (예: 'subject_01/action_01')
        create_dirs (bool): True일 경우 정의된 디렉토리들을 실제로 생성합니다.
    """
    # 기본 데이터 루트 경로 설정
    BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data") # 최상위 데이터 디렉토리 설정

    # 각 단계별 경로 정의 (Path 객체 활용)
    paths = {
        "frame":          BASE_DIR / "1_FRAME" / common_path,                # 추출된 프레임 저장 위치
        "keypoint":       BASE_DIR / "2_KEYPOINTS" / common_path,            # 원본 키포인트(JSON) 저장 위치
        "mp4":            BASE_DIR / "3_MP4" / f"{common_path}.mp4",         # 원본 키포인트 기반 시각화 영상 파일 경로
        "interp_data":    BASE_DIR / "4_INTERP_DATA" / common_path,          # 보간(Interpolation) 처리 후 데이터 저장 위치
        "yolo_txt":       BASE_DIR / "5_YOLO_TXT" / common_path,             # YOLO 학습용 .txt 라벨 저장 위치
        "yolo_dataset":   BASE_DIR / "6_YOLO_TRAINING_DATA" / common_path,   # 최종 데이터셋 구성 위치
        "interp_mp4":     BASE_DIR / "7_INTERP_MP4" / common_path,           # 보간 데이터 기반 시각화 영상 저장 위치
        "sam":            BASE_DIR / "8_SAM" / common_path,                  # SAM 세그멘테이션 결과 저장 위치
        "test":           BASE_DIR / "test" / common_path                    # 임시 테스트 및 결과 확인용 위치
    }

    # 디렉토리 생성 로직 (파일인 .mp4 경로는 제외하고 생성)
    if create_dirs:
        for key, path in paths.items(): # 모든 경로 순회
            if key != "mp4": # mp4는 파일 경로이므로 부모 디렉토리까지만 생성
                path.mkdir(parents=True, exist_ok=True) # 폴더가 없으면 상위 폴더까지 포함하여 생성
            else:
                path.parent.mkdir(parents=True, exist_ok=True) # 파일이 저장될 부모 폴더 생성

    return paths # 모든 경로가 담긴 딕셔너리 반환