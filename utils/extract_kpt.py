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

import json # JSON 파일 입출력을 위한 모듈입니다.
from pathlib import Path # 파일 경로를 객체 지향적으로 다루기 위한 모듈입니다.
import numpy as np # 수치 배열 처리를 위한 넘파이입니다.
from typing import Union # 타입 힌트를 위해 가져옵니다.

def save_12kpt_to_17kpt_json(
    src_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    kpt_array: np.ndarray, 
    target_id: int
) -> bool:
    """
    (N, 12, 3) 형태의 키포인트 배열을 원본 17-Kpt JSON 파일의 5~16번 관절에 병합하여 새로운 폴더에 저장합니다.
    0~4번(얼굴) 데이터는 원본 그대로 훼손 없이 보존됩니다.
    
    Args:
        src_dir: 원본 JSON 파일들이 있는 디렉토리 경로
        output_dir: 수정된 JSON 파일들을 저장할 대상 디렉토리 경로
        kpt_array: 칼만 필터 등이 적용된 (N, 12, 3) 형태의 Numpy 배열
        target_id: JSON 내에서 좌표를 업데이트할 대상의 고유 ID
    """
    src_path = Path(src_dir) # 입력받은 원본 문자열 경로를 Path 객체로 변환합니다.
    dst_path = Path(output_dir) # 입력받은 저장 문자열 경로를 Path 객체로 변환합니다.
    
    # 1. 경로 유효성 검사 및 저장 폴더 생성
    if not src_path.exists():
        print(f"❌ 원본 경로가 존재하지 않습니다: {src_path}") # 에러 메시지를 출력합니다.
        return False
        
    dst_path.mkdir(parents=True, exist_ok=True) # 저장할 폴더가 없다면 새로 생성합니다.
    
    # 2. 파일 리스트 확보 및 정렬
    json_files = sorted(list(src_path.glob("*.json"))) # 프레임 순서대로 처리하기 위해 파일명을 정렬합니다.
    
    if not json_files:
        print("⚠️ 처리할 JSON 파일이 폴더에 없습니다.") # 빈 폴더일 경우의 예외 처리입니다.
        return False
        
    # 원본 파일 개수와 넘파이 배열의 프레임 수가 다르면 짝이 맞지 않으므로 경고를 띄워줍니다.
    if len(json_files) != len(kpt_array):
        print(f"⚠️ 경고: JSON 파일 개수({len(json_files)})와 배열의 프레임 수({len(kpt_array)})가 다릅니다!")

    print(f"\n🚀 12-Kpt 데이터를 원본 JSON(17-Kpt)에 병합하여 저장합니다...")
    print(f"📂 저장 경로: {dst_path}")
    
    processed_count = 0 # 성공적으로 저장된 파일 개수를 추적합니다.
    
    # 3. 파일과 배열을 1:1로 매칭하여 병합 및 저장 수행
    for json_file, kpt_frame in zip(json_files, kpt_array): # zip을 통해 파일과 프레임 배열을 순서대로 하나씩 꺼냅니다.
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f) # 원본 JSON 데이터를 파이썬 딕셔너리로 읽어옵니다.
                
            for inst in data.get('instance_info', []): # 인식된 사람(인스턴스) 목록을 순회합니다.
                inst_id = inst.get('instance_id') if inst.get('instance_id') is not None else inst.get('id')
                
                if inst_id == target_id: # 우리가 찾고자 하는 ID와 일치할 때만 덮어씁니다.
                    
                    # 🌟 12개의 관절 데이터를 원본 JSON의 5~16번 인덱스에 병합합니다.
                    for i in range(12): 
                        json_idx = i + 5 # kpt_array의 0번 인덱스는 JSON의 5번(어깨) 인덱스에 해당합니다.
                        
                        inst['keypoints'][json_idx][0] = float(kpt_frame[i, 0]) # X 좌표를 실수형으로 업데이트합니다.
                        inst['keypoints'][json_idx][1] = float(kpt_frame[i, 1]) # Y 좌표를 실수형으로 업데이트합니다.
                        
                        # Score 형태가 단일 값인지 리스트인지 구분하여 안전하게 업데이트합니다.
                        if isinstance(inst['keypoint_scores'][json_idx], list):
                            inst['keypoint_scores'][json_idx][0] = float(kpt_frame[i, 2])
                        else:
                            inst['keypoint_scores'][json_idx] = float(kpt_frame[i, 2])
                            
                    break # 타겟 ID의 좌표를 모두 수정했으므로, 더 이상 찾지 않고 루프를 탈출합니다.
                    
            # 4. 수정된 데이터를 새 파일로 저장
            save_path = dst_path / json_file.name # 원본 파일명과 동일하게 새 경로를 만듭니다.
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4) # 사람이 읽을 수 있도록 예쁘게 들여쓰기(indent)하여 저장합니다.
                
            processed_count += 1 # 저장이 완료되면 카운트를 증가시킵니다.
            
        except Exception as e:
            print(f"⚠️ {json_file.name} 처리 중 에러 발생: {e}") # 에러가 나더라도 전체 프로세스가 죽지 않도록 잡아줍니다.
            
    print(f"✅ 완료: 총 {processed_count}개의 파일이 성공적으로 저장되었습니다!\n")
    return True # 전체 과정이 정상적으로 끝났음을 반환합니다.