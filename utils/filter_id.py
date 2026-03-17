import json # JSON 파일을 다루기 위해 파이썬 내장 라이브러리를 불러옵니다.
from pathlib import Path # 파일 및 폴더 경로를 안전하게 다루기 위해 Path 모듈을 불러옵니다.
from typing import List, Union, Optional # 타입 힌트를 위해 필요한 모듈들을 불러옵니다.
from tqdm import tqdm # 터미널에 예쁜 진행률 표시줄을 띄우기 위해 tqdm을 불러옵니다.

def filter_skeleton_by_ids(
    input_path: Union[str, Path], 
    output_path: Union[str, Path], 
    target_ids: List[int],
    start_idx: int = 0, # 🌟 변수명을 직관적으로 start_idx에서 start_frame으로 변경했습니다.
    end_idx: Optional[int] = None # 🌟 end_idx 역시 end_frame으로 변경했습니다.
): 
    """폴더 내의 JSON 파일명(프레임 번호)을 기준으로 특정 범위의 파일만 필터링합니다."""
    
    in_dir = Path(input_path) # 문자열로 들어올 수 있는 입력 경로를 Path 객체로 변환합니다.
    out_dir = Path(output_path) # 문자열 출력 경로 역시 Path 객체로 변환하여 다룹니다.
    
    if not in_dir.exists(): # 입력 폴더가 존재하지 않는지 검사합니다.
        print(f"❌ 오류: 입력 폴더를 찾을 수 없습니다. ({in_dir})") # 에러 상황을 알립니다.
        return # 함수 실행을 즉시 중단하여 추가적인 오류를 막습니다.

    out_dir.mkdir(parents=True, exist_ok=True) # 출력할 폴더를 안전하게 생성합니다. (이미 있어도 통과)
    
    target_id_set = set(int(id_) for id_ in target_ids) # 탐색 속도 향상을 위해 타겟 ID 리스트를 Set 구조로 변환합니다.
    
    # 🌟 1. 폴더 내 모든 JSON 파일을 가져와 오름차순으로 정렬합니다.
    json_files = sorted(list(in_dir.glob("*.json"))) # 확장자가 json인 파일들을 찾아 이름순 정렬합니다.
    
    # 🌟 2. 파일명에서 숫자를 추출하여 원하는 범위의 파일만 걸러냅니다.
    target_files = [] # 조건에 맞는 파일들만 담을 빈 리스트를 준비합니다.
    for json_file in json_files: # 폴더 안의 모든 파일을 하나씩 확인합니다.
        try:
            # json_file.stem은 확장자를 제외한 파일명만 가져옵니다. (예: "025870.json" -> "025870")
            frame_num = int(json_file.stem) # 파일명을 정수형 숫자로 변환합니다.
        except ValueError: # 만약 파일명이 "result.json" 처럼 숫자가 아니라면,
            continue # 에러를 내지 않고 해당 파일은 부드럽게 건너뜁니다.
            
        # 현재 파일의 프레임 번호가 시작 프레임보다 크거나 같아야 합니다.
        is_after_start = start_idx <= frame_num # 시작 조건 만족 여부를 확인합니다.
        
        # 종료 프레임이 지정되지 않았거나(None), 현재 프레임이 종료 프레임보다 작거나 같아야 합니다.
        is_before_end = (end_idx is None) or (frame_num <= end_idx) # 종료 조건 만족 여부를 확인합니다.
        
        if is_after_start and is_before_end: # 두 조건을 모두 만족한다면 (우리가 원하는 범위라면),
            target_files.append(json_file) # 해당 파일을 타겟 리스트에 추가합니다.
            
    # 처리할 최종 파일 개수와 범위를 출력하여 다시 한번 확인합니다.
    print(f"🚀 총 {len(target_files)}개의 JSON 파일을 처리합니다. (범위: {start_idx} ~ {end_idx if end_idx else '끝'}, 남길 ID: {target_id_set})\n") 
    
    # 🌟 3. 걸러진 파일들을 대상으로 필터링 작업을 시작합니다.
    for json_file in tqdm(target_files, desc="JSON 필터링 진행 중"): # 타겟 파일들을 진행 표시줄과 함께 순회합니다.
        with open(json_file, 'r', encoding='utf-8') as f: # 원본 JSON 파일을 한글 깨짐 없이 읽기 모드로 엽니다.
            data = json.load(f) # JSON 파일의 내용을 파이썬 딕셔너리로 불러옵니다.
            
        if "instance_info" in data: # 데이터 내부에 사람(스켈레톤) 정보가 있는지 확인합니다.
            filtered_instances = [] # 우리가 원하는 ID의 사람만 담을 빈 리스트입니다.
            
            for instance in data["instance_info"]: # 검출된 모든 사람의 정보를 하나씩 확인합니다.
                current_id = instance.get("instance_id") # 현재 사람의 고유 ID를 가져옵니다.
                if current_id in target_id_set: # 그 ID가 우리가 남기려는 타겟 ID에 속한다면,
                    filtered_instances.append(instance) # 보존할 리스트에 안전하게 담습니다.
                    
            data["instance_info"] = filtered_instances # 원본 데이터를 우리가 걸러낸 타겟 정보로만 덮어씌웁니다.
            
        save_path = out_dir / json_file.name # 출력 폴더에 동일한 파일명으로 저장 경로를 지정합니다.
        with open(save_path, 'w', encoding='utf-8') as f: # 새롭게 저장할 파일을 쓰기 모드로 엽니다.
            json.dump(data, f, indent=4) # 보기 좋게 들여쓰기하여 JSON 데이터로 저장합니다.
            
    print(f"\n✅ 완료! 필터링된 파일들이 '{out_dir}' 폴더에 성공적으로 저장되었습니다.") # 최종 성공 메시지를 띄웁니다.