import json # JSON 처리를 위한 라이브러리입니다.
import numpy as np # 수치 연산 및 배열 생성을 위한 넘파이입니다.
from pathlib import Path # 경로 처리를 위한 라이브러리입니다.
from tqdm import tqdm # 진행 상태 확인을 위한 라이브러리입니다.
from typing import Union # 타입 힌트를 통해 코드의 안정성을 높이는 라이브러리입니다.


# ==========================================
# 함수 1: 원본 데이터 추출 (Extraction) - 구간(Segment) 지원 버전
# ==========================================
def extract_id_keypoints(
    json_dir: Union[str, Path],     # JSON 파일들이 있는 디렉토리 경로를 받습니다.
    target_id: int,                 # 추출하고자 하는 특정 사람의 ID 번호입니다.
    start_frame: int = 0,           # 추출을 시작할 프레임 번호의 기본값을 0으로 설정합니다.
    end_frame: float = float('inf') # 추출을 종료할 프레임 번호의 기본값을 무한대(끝까지)로 설정합니다.
) -> np.ndarray:                    # 이 함수가 최종적으로 넘파이 배열을 반환함을 명시합니다.
    """
    JSON 디렉토리에서 특정 ID의 [x, y, score]를 추출하여 (N, 12, 3) 배열을 반환합니다.
    지정된 start_frame과 end_frame 구간 내의 파일만 처리하여 실행 속도를 최적화했습니다.
    """
    json_path = Path(json_dir) # 문자열 경로를 Path 객체로 변환하여 파일 시스템을 안전하게 다룹니다.
    all_json_files = sorted(list(json_path.glob("*.json"))) # 폴더 내 모든 JSON 파일을 이름순(프레임순)으로 정렬하여 리스트로 만듭니다.
    
    target_files = [] # 설정한 프레임 구간에 포함되는 파일들만 따로 모아둘 빈 리스트입니다.
    
    # 💡 [성능 개선] 전체 파일을 무조건 열지 않고, 파일명으로 먼저 필터링하여 불필요한 파일 접근(I/O)을 방지합니다.
    for f in all_json_files: # 찾은 모든 JSON 파일에 대해 반복합니다.
        try:
            frame_idx = int(f.stem) # 파일명(확장자 제외)을 정수형 프레임 번호로 변환합니다. (예: '000123' -> 123)
            if start_frame <= frame_idx <= end_frame: # 현재 프레임 번호가 설정한 구간 안에 포함되는지 검사합니다.
                target_files.append(f) # 조건에 맞으면 실제 처리할 타겟 리스트에 추가합니다.
        except ValueError: # 파일명이 숫자로 변환되지 않는 경우(예: 'config.json')를 안전하게 건너뜁니다.
            continue 
    
    raw_data = [] # 추출된 프레임별 키포인트 데이터를 순서대로 차곡차곡 저장할 리스트입니다.

    # 필터링된 파일들만 대상으로 진행바와 함께 순회합니다. 사용자에게 현재 어떤 구간을 처리 중인지 보여줍니다.
    for file in tqdm(target_files, desc=f"Extracting ID:{target_id} ({start_frame}~{end_frame})"): 
        with open(file, 'r', encoding='utf-8') as f: # 한글 등의 문자가 깨지지 않도록 utf-8 인코딩으로 파일을 엽니다.
            data = json.load(f) # JSON 파일의 내용을 파이썬 딕셔너리로 변환하여 메모리에 올립니다.
        
        # ID가 아예 없거나 화면에서 사라진 프레임을 대비해 0으로 채워진 기본 배열을 만듭니다. (12개 관절, 3개 값)
        frame_data = np.zeros((12, 3)) 
        
        for inst in data.get('instance_info', []): # JSON 내의 '인식된 모든 사람 정보'를 하나씩 확인합니다.
            # 현재 확인 중인 사람의 ID가 우리가 찾고 있는 target_id와 일치하는지 검사합니다.
            if inst.get('instance_id') == target_id or inst.get('id') == target_id: 
                
                coords = np.array(inst.get('keypoints', [])) # [17, 2] 형태의 (x, y) 좌표 리스트를 넘파이 배열로 변환합니다.
                scores = np.array(inst.get('keypoint_scores', [])).reshape(-1, 1) # [17, 1] 형태로 신뢰도 점수를 세로 배열로 맞춥니다.
                
                # 데이터가 비정상적으로 적지 않은지(17개 정상 유무) 체크하는 안전망을 추가했습니다.
                if len(coords) >= 17 and len(scores) >= 17: 
                    full_kpts = np.hstack([coords[:, :2], scores]) # 좌표(x, y)와 점수(score)를 가로(hstack)로 이어붙여 [17, 3] 배열을 만듭니다.
                    
                    # 전체 17개 관절 중 얼굴(0~4번)을 제외하고 신체 관절인 5번부터 16번까지 총 12개를 슬라이싱합니다.
                    frame_data = full_kpts[5:17, :] 
                    
                break # 원하는 ID의 데이터를 성공적으로 추출했으므로, 다른 사람 데이터는 볼 필요 없이 내부 루프를 종료합니다.
        
        raw_data.append(frame_data) # 방금 추출한 [12, 3] 데이터를 전체 결과를 담는 리스트에 추가합니다.

    return np.array(raw_data) # 리스트에 쌓인 모든 프레임 데이터를 (Frames, 12, 3) 형태의 3차원 넘파이 배열로 묶어 반환합니다.

# ==========================================
# 함수 2: 데이터 정규화 (12 Keypoints 대응)
# ==========================================
def normalize_skeleton_array(data_array):
    """
    (N, 12, 3) 형태의 배열을 받아 정규화합니다.
    (인덱스 주의: 원래 5-16번이 0-11번으로 당겨짐)
    """
    norm_data = data_array.copy().astype(float)
    
    for i in range(len(norm_data)):
        kpts = norm_data[i] # 현재 프레임 (12, 3)
        
        if np.all(kpts == 0):
            continue
        
        # 1. 중앙점 계산: 골반 중심 (6번, 7번 중점)
        hip_center = (kpts[6, :2] + kpts[7, :2]) / 2.0
        
        # 2. 기준 거리 계산: 몸통 길이 (어깨 중점과 골반 중점 사이 거리)
        shoulder_center = (kpts[0, :2] + kpts[1, :2]) / 2.0
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        if torso_length > 1e-6:
            norm_data[i, :, :2] = (kpts[:, :2] - hip_center) / torso_length
            
    return norm_data

# ==========================================
# 함수 3: 12kpt 17kpt와 동일한 형식으로 덮어쓰기 (다른 ID 유지)
# ==========================================
def save_12kpt_to_17kpt_json(
    src_dir, 
    output_dir, 
    kpt_array, 
    target_id,
    start_frame=0  # 리스트 상의 시작 인덱스로 활용됩니다.
):
    src_path = Path(src_dir) # 소스 디렉토리 경로 객체를 생성합니다.
    dst_path = Path(output_dir) # 출력 디렉토리 경로 객체를 생성합니다.
    dst_path.mkdir(parents=True, exist_ok=True) # 출력 폴더가 없다면 안전하게 생성합니다.

    # 🌟 원본 디렉토리의 모든 JSON 파일을 가져와 이름순으로 정렬합니다.
    json_files = sorted(src_path.glob('*.json')) # 파일명 기반으로 오름차순 정렬을 보장합니다.
    
    # 방어 로직: 시작 프레임이 전체 파일 수보다 크면 중단합니다.
    if start_frame >= len(json_files): # 인덱스 초과 오류를 방지합니다.
        print(f"⚠️ 에러: 시작 프레임({start_frame})이 전체 파일 수({len(json_files)})를 벗어납니다.") # 경고 메시지를 출력합니다.
        return # 함수 실행을 안전하게 종료합니다.

    start_file_name = json_files[start_frame].name # 시작 지점의 실제 파일명을 추출합니다.
    print(f"🚀 {start_file_name} 파일부터 병합을 시작합니다...") # 사용자에게 시작 지점을 알립니다.

    processed_count = 0 # 성공적으로 처리된 파일 개수를 추적합니다.
    
    # kpt_array의 길이만큼 반복하며 매칭되는 JSON을 업데이트합니다.
    for i in range(len(kpt_array)): # 배열 데이터만큼 루프를 돕니다.
        file_idx = start_frame + i # 실제 처리할 JSON 파일의 리스트 인덱스를 계산합니다.
        
        # 배열은 남았는데 파일이 부족할 경우를 대비한 안전장치입니다.
        if file_idx >= len(json_files): # 더 이상 읽을 파일이 없다면 중단합니다.
            print(f"⚠️ 매칭할 원본 파일이 부족하여 루프를 조기 종료합니다. (초과 인덱스: {file_idx})") # 로그를 남깁니다.
            break # 반복문을 안전하게 빠져나갑니다.
            
        json_file = json_files[file_idx] # 처리할 원본 JSON 파일의 전체 경로입니다.
        json_name = json_file.name # 'frame_001.json' 등 원본 파일의 정확한 이름을 가져옵니다.
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f: # 파일을 읽기 모드로 엽니다.
                data = json.load(f) # JSON 데이터를 파이썬 딕셔너리로 변환합니다.

            found_target = False # 타겟 ID를 찾았는지 여부를 기록하는 플래그입니다.
            
            for inst in data.get('instance_info', []): # instance_info 리스트를 순회합니다.
                # 'instance_id' 키가 없으면 'id' 키를 안전하게 가져옵니다.
                inst_id = inst.get('instance_id') if inst.get('instance_id') is not None else inst.get('id') 
                
                if inst_id == target_id: # 찾고자 하는 타겟 ID와 일치한다면 업데이트를 시작합니다.
                    for k in range(12): # 12개의 새로운 키포인트를 반복합니다.
                        json_idx = k + 5 # 17개 키포인트 포맷에 맞추기 위해 5칸 오프셋을 적용합니다.
                        # kpt_array의 x, y 좌표를 float 형태로 덮어씁니다.
                        inst['keypoints'][json_idx][0] = float(kpt_array[i, k, 0]) 
                        inst['keypoints'][json_idx][1] = float(kpt_array[i, k, 1]) 
                    found_target = True # 대상을 찾아 업데이트했음을 표시합니다.
                    break # 현재 파일에서 대상 수정을 완료했으므로 내부 루프를 종료합니다.
            
            # 🌟 원본과 완전히 동일한 이름(json_name)으로 새 디렉토리에 저장합니다.
            save_path = dst_path / json_name # 저장할 최종 경로를 구성합니다.
            with open(save_path, 'w', encoding='utf-8') as f: # 쓰기 모드로 파일을 엽니다.
                json.dump(data, f, indent=4) # 수정된 딕셔너리를 보기 좋게(indent=4) 저장합니다.
            
            processed_count += 1 # 처리 카운트를 1 증가시킵니다.

        except Exception as e: # 예상치 못한 파일 읽기/쓰기 오류를 잡습니다.
            print(f"⚠️ {json_name} 처리 중 에러 발생: {e}") # 어떤 파일에서 에러가 났는지 출력합니다.

    # 루프 종료 후 최종 결과를 요약하여 출력합니다.
    last_file_name = json_files[start_frame + processed_count - 1].name # 마지막으로 저장된 파일의 이름을 확인합니다.
    print(f"✅ 완료: 총 {processed_count}개 파일 변환 완료! (종료 파일: {last_file_name})") # 성공 알림을 띄웁니다.

# ==========================================
# 함수 4: 12kpt 17kpt와 동일한 형식으로 덮어쓰기 (patient_id 만 유지)
# ==========================================

def save_patient_only_12_to_17(
    src_dir,            # 원본 JSON 파일들이 위치한 디렉토리 경로입니다.
    output_dir,         # 수정된 JSON 파일들을 저장할 디렉토리 경로입니다.
    kpt_array,          # 덮어쓸 12개의 새로운 키포인트 좌표 배열입니다.
    patient_id,         # 필터링 및 업데이트할 대상 환자의 ID (기존 target_id)입니다.
    start_frame=0       # 리스트 상의 시작 인덱스로 활용됩니다 (기본값 0).
):
    src_path = Path(src_dir) # 소스 디렉토리 문자열을 Path 객체로 변환합니다.
    dst_path = Path(output_dir) # 출력 디렉토리 문자열을 Path 객체로 변환합니다.
    dst_path.mkdir(parents=True, exist_ok=True) # 출력 폴더가 없다면 부모 폴더를 포함하여 안전하게 생성합니다.

    json_files = sorted(src_path.glob('*.json')) # 원본 디렉토리의 모든 JSON 파일을 찾아 이름순으로 오름차순 정렬합니다.
    
    if start_frame >= len(json_files): # 시작 프레임이 전체 파일 수보다 크거나 같으면 인덱스 에러가 발생하므로 검사합니다.
        print(f"⚠️ 에러: 시작 프레임({start_frame})이 전체 파일 수({len(json_files)})를 벗어납니다.") # 경고 메시지를 출력합니다.
        return # 더 이상 진행할 수 없으므로 함수 실행을 안전하게 종료합니다.

    start_file_name = json_files[start_frame].name # 처리를 시작할 첫 번째 파일의 이름을 가져옵니다.
    print(f"🚀 {start_file_name} 파일부터 병합을 시작합니다...") # 사용자에게 작업 시작을 알리는 로그를 출력합니다.

    processed_count = 0 # 성공적으로 변환 및 저장된 파일의 개수를 누적할 변수입니다.
    
    for i in range(len(kpt_array)): # 12kpt 배열 데이터의 길이만큼 순회하며 파일과 매칭합니다.
        file_idx = start_frame + i # 시작 프레임에 현재 반복 횟수를 더해 처리할 실제 파일 인덱스를 구합니다.
        
        if file_idx >= len(json_files): # 인덱스가 파일 리스트 길이를 초과하면 에러를 방지하기 위해 검사합니다.
            print(f"⚠️ 매칭할 원본 파일이 부족하여 루프를 조기 종료합니다. (초과 인덱스: {file_idx})") # 로그를 남겨 원인을 알립니다.
            break # 남은 배열 데이터가 있더라도 반복문을 안전하게 빠져나갑니다.
            
        json_file = json_files[file_idx] # 인덱스를 통해 처리할 원본 JSON 파일의 전체 경로 객체를 가져옵니다.
        json_name = json_file.name # 'frame_001.json' 과 같은 원본 파일명을 문자열로 추출합니다.
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f: # 원본 파일을 읽기 모드로 안전하게 엽니다.
                data = json.load(f) # JSON 파일의 텍스트 데이터를 파이썬 딕셔너리로 변환하여 메모리에 올립니다.

            filtered_instances = [] # 대상 patient_id와 일치하는 스켈레톤 정보만 담기 위해 빈 리스트를 초기화합니다.
            
            for inst in data.get('instance_info', []): # 딕셔너리에서 instance_info 리스트를 가져와 순회합니다[cite: 1].
                inst_id = inst.get('instance_id') if inst.get('instance_id') is not None else inst.get('id') # instance_id가 없으면 id 키를 대신 가져와 할당합니다[cite: 1].
                
                if inst_id == patient_id: # 현재 스켈레톤의 ID가 입력받은 환자 ID와 일치하는지 확인합니다[cite: 1].
                    for k in range(12): # 12개의 새로운 키포인트를 반복하여 기존 데이터를 덮어쓸 준비를 합니다.
                        json_idx = k + 5 # 17개 키포인트 포맷에 맞추기 위해 앞의 5개(얼굴 등)를 건너뛰는 오프셋을 적용합니다.
                        inst['keypoints'][json_idx][0] = float(kpt_array[i, k, 0]) # 3D/2D 배열의 x 좌표를 추출해 float 형태로 키포인트 x값을 덮어씁니다[cite: 1].
                        inst['keypoints'][json_idx][1] = float(kpt_array[i, k, 1]) # 3D/2D 배열의 y 좌표를 추출해 float 형태로 키포인트 y값을 덮어씁니다[cite: 1].
                    
                    filtered_instances.append(inst) # 키포인트 좌표 업데이트가 끝난 해당 인스턴스 딕셔너리를 필터 리스트에 추가합니다.
                    break # 한 이미지 내에 동일한 ID는 한 명뿐이므로, 찾은 즉시 내부 루프를 종료해 성능을 최적화합니다.
            
            data['instance_info'] = filtered_instances # 원본 데이터의 instance_info를 타겟 환자 1명만 들어있는 리스트로 덮어씌웁니다.
            
            save_path = dst_path / json_name # 대상 폴더 경로와 원본 파일명을 결합하여 최종 저장 경로를 생성합니다.
            with open(save_path, 'w', encoding='utf-8') as f: # 필터링된 데이터를 쓰기 모드로 안전하게 엽니다.
                json.dump(data, f, indent=4) # 불필요한 스켈레톤이 제거된 딕셔너리를 가독성 좋게(들여쓰기 4칸) 저장합니다.
            
            processed_count += 1 # 파일 하나의 읽기, 수정, 필터링, 쓰기가 무사히 끝났으므로 처리 카운트를 1 증가시킵니다.

        except Exception as e: # 파일을 읽거나 쓰는 도중 발생할 수 있는 예상치 못한 모든 오류를 잡습니다.
            print(f"⚠️ {json_name} 처리 중 에러 발생: {e}") # 어떤 파일에서 어떤 에러가 발생했는지 콘솔에 출력해 디버깅을 돕습니다.

    if processed_count > 0: # 1개 이상의 파일이 정상적으로 변환 및 저장되었는지 확인합니다.
        last_file_name = json_files[start_frame + processed_count - 1].name # 마지막으로 처리되어 저장된 파일의 이름을 확인합니다.
        print(f"✅ 완료: 총 {processed_count}개 파일 변환 완료! (종료 파일: {last_file_name})") # 사용자에게 성공적인 종료 알림 및 요약 정보를 출력합니다.
    else: # 처리된 파일이 하나도 없는 경우 (예: 조건에 맞는 대상 ID가 전혀 없는 경우)입니다.
        print("⚠️ 완료: 처리된 파일이 없습니다. (조건에 맞는 파일이나 대상 ID가 없을 수 있습니다.)") # 사용자에게 결과가 없었음을 친절하게 안내합니다.