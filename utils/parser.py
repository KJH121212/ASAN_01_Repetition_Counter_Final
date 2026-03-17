from pathlib import Path

def parse_common_path(common_path: str):
    """
    common_path를 분석하여 촬영 각도(camera_angle)와 운동 명칭(exercise_name)을 추출합니다.
    예: AI_dataset/.../diagonal__biceps_curl__1 -> ('diagonal', 'biceps_curl')
    """
    # 1. 경로의 맨 마지막 폴더 이름만 쏙 뽑아냅니다.
    # 예: "diagonal__biceps_curl" 또는 "frontal__biceps_curl__1"
    folder_name = Path(common_path).name 
    
    # 2. 지정된 구분자('__')를 기준으로 문자열을 쪼갭니다.
    parts = folder_name.split('__')
    
    # 3. 데이터가 규칙에 맞게 2개 이상으로 잘 쪼개졌는지 확인합니다.
    if len(parts) >= 2:
        camera_angle = parts[0]     # 첫 번째 조각: 촬영 각도 (예: diagonal)
        exercise_name = parts[1]    # 두 번째 조각: 운동 명칭 (예: biceps_curl)
        
        # 만약 parts[2]에 넘버링('1')이 있더라도 무시하고 필요한 두 가지만 반환합니다.
        return camera_angle, exercise_name
    
    else:
        # 규칙에 맞지 않는 예외 경로일 경우를 대비한 안전망입니다.
        print(f"⚠️ [경고] 경로 이름 규칙이 맞지 않습니다: {folder_name}")
        return None, None