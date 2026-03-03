import json # JSON 파일을 읽고 쓰기 위해 파이썬 내장 json 라이브러리를 불러옵니다.
from pathlib import Path # 폴더 및 파일 경로를 객체지향적이고 안전하게 다루기 위해 Path 모듈을 불러옵니다.
from typing import List, Union # 타입 힌트를 사용하여 함수의 입력값 종류를 명확하게 지정하기 위해 불러옵니다.
from tqdm import tqdm # 🌟 진행 상황을 시각적인 바(bar) 형태로 보여주기 위해 tqdm 라이브러리를 불러옵니다.

def filter_skeleton_by_ids(input_path: Union[str, Path], output_path: Union[str, Path], target_ids: List[int]): # 입력 경로, 출력 경로, 남길 ID 리스트를 받는 함수를 정의합니다.
    """지정된 폴더의 JSON 파일들에서 특정 instance_id를 가진 데이터만 남기고 새 폴더에 저장합니다.""" # 함수의 역할을 설명하는 독스트링(Docstring)입니다.
    
    in_dir = Path(input_path) # 입력받은 문자열 경로를 Path 객체로 변환하여 다루기 쉽게 만듭니다.
    out_dir = Path(output_path) # 입력받은 문자열 출력 경로도 Path 객체로 변환합니다.
    
    if not in_dir.exists(): # 만약 입력 폴더가 실제로 존재하지 않는다면,
        print(f"❌ 오류: 입력 폴더를 찾을 수 없습니다. ({in_dir})") # 에러 메시지를 출력하고,
        return # 함수 실행을 즉시 중단합니다.

    out_dir.mkdir(parents=True, exist_ok=True) # 출력할 폴더가 없다면 상위 폴더까지 한 번에 생성하며, 이미 있어도 에러를 내지 않습니다.
    
    # JSON 파일 내부의 ID가 정수형(int)이므로, 입력받은 target_ids도 비교를 위해 안전하게 정수형 세트(set)로 변환합니다.
    target_id_set = set(int(id_) for id_ in target_ids) # 리스트 탐색보다 세트 탐색이 훨씬 빠르므로(O(1)) 변환해 줍니다.
    
    json_files = list(in_dir.glob("*.json")) # 입력 폴더 내부에 있는 확장자가 '.json'인 모든 파일을 찾아 리스트로 만듭니다.
    print(f"🚀 총 {len(json_files)}개의 JSON 파일을 처리합니다. (남길 ID: {target_id_set})\n") # 작업 시작을 경쾌하게 알립니다.
    
    # 🌟 기존 리스트를 tqdm()으로 감싸서 터미널에 예쁜 진행률 표시줄을 만들어 줍니다! desc 속성으로 제목도 달아주었어요.
    for json_file in tqdm(json_files, desc="JSON 필터링 진행 중"): # 찾은 JSON 파일들을 처음부터 끝까지 진행 바와 함께 하나씩 순회합니다.
        
        # 1. 원본 JSON 파일 읽기
        with open(json_file, 'r', encoding='utf-8') as f: # 한글 깨짐 방지를 위해 utf-8 인코딩으로 파일을 엽니다.
            data = json.load(f) # 파일 내용 전체를 파이썬의 딕셔너리 형태로 메모리에 불러옵니다.
            
        # 2. 필터링 로직 적용
        if "instance_info" in data: # 불러온 데이터 안에 'instance_info'라는 키가 정상적으로 존재하는지 확인합니다.
            filtered_instances = [] # 우리가 남기고 싶은 타겟 데이터만 골라 담을 새로운 빈 리스트를 준비합니다.
            
            for instance in data["instance_info"]: # 검출된 모든 사람(스켈레톤) 데이터를 하나씩 꺼내어 검사합니다.
                current_id = instance.get("instance_id") # 현재 스켈레톤 데이터가 가지고 있는 고유 ID 값을 안전하게 가져옵니다.
                
                if current_id in target_id_set: # 만약 그 ID가 우리가 남기고자 하는 타겟 ID 목록 안에 포함되어 있다면,
                    filtered_instances.append(instance) # 준비해둔 새로운 리스트에 해당 스켈레톤 데이터를 통째로 추가(보존)합니다.
                    
            data["instance_info"] = filtered_instances # 기존에 여러 명이 있던 리스트를, 방금 걸러낸 타겟들만 있는 리스트로 덮어씌웁니다.
            
        # 3. 새로운 경로에 저장
        save_path = out_dir / json_file.name # 출력 폴더 경로에 현재 작업 중인 파일의 이름(예: 000002.json)을 합쳐 최종 저장 경로를 만듭니다.
        with open(save_path, 'w', encoding='utf-8') as f: # 수정된 데이터를 저장하기 위해 쓰기('w') 모드로 파일을 엽니다.
            json.dump(data, f, indent=4) # 사람이 읽기 편하도록 들여쓰기(indent=4)를 적용하여 JSON 파일로 깔끔하게 저장합니다.
            
    print(f"\n✅ 완료! 필터링된 파일들이 '{out_dir}' 폴더에 성공적으로 저장되었습니다.") # 모든 반복문이 끝나면 성공 메시지를 출력합니다. (줄바꿈 \n 추가)