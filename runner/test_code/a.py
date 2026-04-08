import os # 파일 시스템 조작을 위한 내장 모듈을 불러옵니다.
import shutil # 파일 이동(move)을 위한 내장 모듈을 불러옵니다.
from pathlib import Path # 안전하고 직관적인 경로 처리를 위해 Path 객체를 불러옵니다.
import pandas as pd # 데이터프레임 처리를 위해 pandas 라이브러리를 불러옵니다.
import sys # 파이썬 인터프리터가 제공하는 변수와 함수를 제어하기 위한 모듈을 불러옵니다.
from tqdm import tqdm # 터미널에 진행률 표시줄(Progress bar)을 띄우기 위해 tqdm 모듈을 불러옵니다.

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/") # 프로젝트의 최상위 기준 경로를 설정합니다.
sys.path.append(str(BASE_DIR)) # 사용자 정의 모듈을 인식할 수 있도록 시스템 경로에 BASE_DIR을 추가합니다.
from utils.path_list_d03 import path_list_d03 # 외부 유틸리티 파일에서 경로 리스트 생성 함수를 불러옵니다.

def organize_files_strictly(paths_dict: dict, limit: int = 10000): # 경로 딕셔너리와 최대 파일 개수 제한을 인자로 받는 함수를 정의합니다.
    """
    미리 생성된 경로 딕셔너리를 전달받아, 파일들을 하위 폴더('01', '02'...)로 정리합니다.
    tqdm을 활용하여 파일 이동 진행률을 시각적으로 제공합니다.
    """
    exclude_keys = ["mp4", "interp_mp4"] # 파일 분할 대상에서 구조를 유지해야 하는 영상 폴더 키워드를 리스트로 정의합니다.

    for key, folder_path in paths_dict.items(): # 전달받은 딕셔너리에서 키(폴더명)와 실제 경로를 순차적으로 하나씩 꺼냅니다.
        if key in exclude_keys: # 현재 처리하려는 폴더가 제외 대상(exclude_keys)에 포함되는지 검사합니다.
            continue # 제외 대상이라면 아무런 작업 없이 다음 루프(폴더)로 넘어갑니다.

        if not folder_path.is_dir(): # 해당 경로가 실제로 시스템상에 디렉토리로 존재하는지 확인합니다.
            print(f"Warning: {folder_path} 경로가 존재하지 않아 건너뜁니다.") # 존재하지 않는다면 사용자에게 경고 메시지를 출력합니다.
            continue # 존재하지 않는 폴더이므로 다음 경로 확인으로 넘어갑니다.

        files = sorted([f for f in folder_path.iterdir() if f.is_file()]) # 해당 폴더 내부의 파일만 찾아 파일명 기준으로 오름차순 정렬하여 리스트를 만듭니다.
        
        if not files: # 추출한 파일 리스트가 비어있는지(이동할 파일이 아예 없는지) 확인합니다.
            print(f"Pass: {key} (이동할 파일이 존재하지 않습니다.)") # 처리할 파일이 없다는 안내 메시지를 화면에 출력합니다.
            continue # 다음 폴더를 처리하기 위해 루프를 건너뜁니다.

        print(f"\nProcessing: {key} (총 {len(files)}개 파일을 하위 폴더로 정리 중...)") # 본격적인 작업 시작 전 처리할 파일의 총 개수를 알립니다.

        # tqdm 컨텍스트 매니저를 시작하여 진행률 표시줄 객체(pbar)를 생성합니다.
        with tqdm(total=len(files), desc=f"{key} 정리", unit="file") as pbar: # 전체 파일 수를 총합으로 설정하고 프로그레스 바를 렌더링합니다.
            for i in range(0, len(files), limit): # 0부터 전체 파일 수까지 사용자가 설정한 한계치(limit) 간격으로 루프를 돕니다.
                chunk = files[i : i + limit] # 지정된 limit 개수만큼 파일 리스트를 잘라 하나의 뭉치(chunk)로 만듭니다.
                sub_folder_name = f"{ (i // limit) + 1 :02d}" # 현재 인덱스를 바탕으로 '01', '02' 형태의 하위 폴더 이름을 계산합니다.
                sub_folder_path = folder_path / sub_folder_name # 기존 경로에 계산된 하위 폴더 이름을 붙여 최종 도착지 경로를 만듭니다.
                
                sub_folder_path.mkdir(parents=True, exist_ok=True) # 파일이 들어갈 하위 폴더를 안전하게 생성합니다. (이미 있어도 오류 무시)

                for file in chunk: # 나누어진 파일 뭉치(chunk) 안에서 파일을 하나씩 꺼내어 이동 로직을 수행합니다.
                    try: # 파일 이동 과정에서 발생할 수 있는 예상치 못한 권한/시스템 오류를 대비합니다.
                        shutil.move(str(file), str(sub_folder_path / file.name)) # 원본 위치의 파일을 새로 생성된 하위 폴더 내부로 완전히 이동시킵니다.
                    except Exception as e: # 파일 이동에 실패했을 경우 해당 예외 객체를 e로 받아옵니다.
                        tqdm.write(f"Error moving {file.name}: {e}") # 진행률 표시줄이 깨지지 않도록 일반 print 대신 tqdm.write를 사용하여 에러를 출력합니다.
                    finally: # 에러가 발생했든 정상 이동했든 상관없이 항상 실행되는 블록입니다.
                        pbar.update(1) # 파일 하나에 대한 처리가 끝났으므로 진행률 표시줄의 카운트를 1만큼 증가시킵니다.

        print(f"Completed: {key} 폴더 내 {len(files)}개 파일 정리 완료!") # 하나의 주요 폴더에 대한 모든 작업과 진행률 표시가 끝났음을 알립니다.
if __name__ == "__main__":
    # 1. 경로 설정
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data_d03") # 데이터의 최상위 루트 디렉토리를 설정합니다.
    METADATA_PATH = DATA_DIR / "metadata_v1.0.csv" # 데이터셋 정보가 담긴 메타데이터 파일 경로를 지정합니다.

    # =================================================================
    # 2. 데이터 로드
    # =================================================================
    if not METADATA_PATH.exists(): # 메타데이터 파일이 실제로 존재하는지 확인합니다.
        print(f"❌ 에러: {METADATA_PATH} 파일을 찾을 수 없습니다.") # 파일이 없으면 에러를 출력합니다.
        sys.exit(1) # 프로그램을 비정상 종료합니다.

    print(f"📂 CSV 로드 중... ({METADATA_PATH.name})") # 로드 시작을 사용자에게 알립니다.
    meta_df = pd.read_csv(METADATA_PATH) # CSV 파일을 읽어 데이터프레임으로 변환합니다.
    total_targets = len(meta_df) # 전체 처리해야 할 데이터 행(Row)의 개수를 파악합니다.

    print(f"🚀 총 {total_targets}개의 타겟 데이터를 순차적으로 처리합니다.") # 전체 작업 규모를 알립니다.

    # =================================================================
    # 3. 모든 Row를 순회하며 작업 수행 (Target 0부터 끝까지)
    # =================================================================
    for idx, row in meta_df.iterrows(): # 데이터프레임의 모든 행을 하나씩 순회합니다.
        common_path = row['common_path'] # 현재 행에서 공통 경로 정보를 가져옵니다.
        
        print(f"\n" + "="*60) # 각 타겟별 구분을 위한 구분선을 출력합니다.
        print(f"📦 [Target {idx + 1} / {total_targets}] 처리 시작: {common_path}") # 현재 진행 단계를 표시합니다.
        print("="*60)

        # 현재 common_path에 해당하는 경로 딕셔너리를 생성합니다.
        paths = path_list_d03(common_path) # 1_FRAME부터 test까지의 실제 경로들을 계산합니다.

        # 4. 파일 정리 함수 실행
        # (기존 organize_files_strictly 함수 내의 tqdm 부분을 일반 print로 변경해야 합니다.)
        organize_files_strictly(paths) # 계산된 경로들을 바탕으로 하위 폴더 분할 작업을 시작합니다.

    print(f"\n✅ 모든 작업이 완료되었습니다! (총 {total_targets}개 타겟 완료)") # 모든 반복문이 종료되었음을 알립니다.