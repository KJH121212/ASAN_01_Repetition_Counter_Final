import sys # 시스템 경로 조작을 위해 sys 모듈을 불러옵니다.
import pandas as pd # 데이터프레임 병합, 조작, 검색을 위해 pandas를 불러옵니다.
from pathlib import Path # 파일 및 디렉토리 경로를 객체 지향적으로 안전하게 다루기 위해 사용합니다.
from tqdm import tqdm # 콘솔 창에 반복문의 진행 상황을 시각적인 바(Progress Bar) 형태로 보여줍니다.
import cv2 # 프레임 이미지의 해상도(너비, 높이)를 읽어오기 위해 OpenCV 라이브러리를 불러옵니다.

# =================================================================
# 1. 설정 및 커스텀 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/") # 프로젝트의 최상위 기준 폴더 경로를 설정합니다.
sys.path.append(str(BASE_DIR)) # 파이썬이 사용자 정의 모듈을 인식할 수 있도록 시스템 경로에 프로젝트 경로를 추가합니다.

from utils.path_list import path_list # 입력된 common_path를 기반으로 관련 하위 폴더 경로들을 모두 생성해주는 함수를 불러옵니다.
from yolo.step1_dataset_builder import convert_single_instance_to_yolo # 단일 JSON을 읽어 YOLO 형식(TXT)으로 변환하는 핵심 함수를 불러옵니다.

if __name__ == "__main__": # 이 파이썬 파일이 직접 실행될 때만 아래의 메인 로직이 작동하도록 보호합니다.
    # =================================================================
    # 2. 경로 설정 및 데이터 로드
    # =================================================================
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data") # 원본 데이터와 라벨이 위치한 최상단 베이스 폴더입니다.
    META_CSV_PATH = DATA_DIR / "metadata_v2.1.csv" # 전체 일반 영상에 대한 정보가 담긴 메타데이터 CSV 파일 경로입니다.
    BOSAN_CSV_PATH = DATA_DIR / "bosanjin_seg_data_v2.1.csv" # 정교한 동작 구간(Segment) 정보가 포함된 Bosanjin CSV 파일 경로입니다.

    print(f"📂 데이터 로드 중... ({META_CSV_PATH.name}, {BOSAN_CSV_PATH.name})") # 사용자에게 데이터 로딩 시작 상태를 안내합니다.
    meta_df = pd.read_csv(META_CSV_PATH) # 일반 메타데이터를 메모리에 데이터프레임 구조로 읽어옵니다.
    bosan_df = pd.read_csv(BOSAN_CSV_PATH) # Bosanjin 구간 데이터를 데이터프레임 구조로 읽어옵니다.

    # =================================================================
    # 3. 데이터프레임 병합 및 필터링 (결측치 및 자료형 오염 완벽 방어)
    # =================================================================
    columns_needed = ['common_path', 'start_frame', 'end_frame', 'is_train', 'is_val'] # 함수 실행에 꼭 필요한 5가지 핵심 열 이름만 리스트로 묶습니다.

    meta_df['start_frame'] = pd.NA # 일반 데이터에는 특정 구간 시작점이 없으므로 결측치(NA)로 열을 초기화합니다.
    meta_df['end_frame'] = pd.NA # 일반 데이터에는 종료점도 없으므로 결측치(NA)로 열을 초기화합니다.
    
    meta_subset = meta_df[columns_needed].copy() # 전체 데이터프레임에서 필요한 열만 쏙 빼서 복사본을 만듭니다.
    bosan_subset = bosan_df[columns_needed].copy() # Bosanjin 데이터프레임에서도 동일하게 필요한 열만 빼서 복사합니다.

    condition_not_bosan = ~meta_subset['common_path'].str.contains('Bosanjin', na=False) # meta_subset에 섞여있는 'Bosanjin' 영상들을 걸러내기 위한 마스크를 생성합니다.
    meta_filtered = meta_subset[condition_not_bosan] # 'Bosanjin'이 완벽하게 제거된 순수 일반 영상 데이터프레임만 남깁니다.

    combined_df = pd.concat([meta_filtered, bosan_subset], ignore_index=True) # 일반 영상과 세밀하게 쪼개진 Bosanjin 구간 영상을 위아래로 깔끔하게 합칩니다.

    # [방어 로직]: CSV 파일 내에 'True '처럼 공백이 섞여 텍스트로 인식되는 현상을 막기 위해 강제로 소문자/공백제거 처리 후 Boolean으로 변환합니다.
    combined_df['is_train'] = combined_df['is_train'].astype(str).str.strip().str.lower() == 'true' # 훈련용 여부를 안전한 순수 True/False로 정규화합니다.
    combined_df['is_val'] = combined_df['is_val'].astype(str).str.strip().str.lower() == 'true' # 검증용 여부도 동일하게 정규화합니다.

    target_df = combined_df[combined_df['is_train'] | combined_df['is_val']].copy() # 최종적으로 학습용이거나 검증용인 데이터만 솎아내어 타겟 데이터프레임을 확정합니다.
    print(f"🎯 처리 대상 시퀀스: 총 {len(target_df)}개 (학습 및 검증용 데이터)") # 최종적으로 병합 및 필터링된 시퀀스의 총개수를 출력합니다.

    # =================================================================
    # 4. 시퀀스 순회 및 YOLO 포맷 라벨 변환 (핵심 로직)
    # =================================================================
    success_total = 0 # 전체 데이터셋에 대해 변환에 성공한 JSON 파일의 총개수를 누적할 카운터 변수입니다.

    for idx, row in tqdm(target_df.iterrows(), total=len(target_df), desc="JSON 라벨 변환 중"): # 타겟 데이터프레임의 모든 행을 순회하며 시각적인 진행률을 띄웁니다.
        common_path = row['common_path'] # 현재 처리 중인 시퀀스의 공통 폴더 기준 경로를 가져옵니다.
        
        # 🌟 사용자 요청 로직: 원본 meta_df에서 common_path가 일치하는 행을 찾아 환자 ID를 안전하게 정수형으로 꺼내옵니다.
        patient_id = int(meta_df.loc[meta_df['common_path'] == common_path, 'patient_id'].iloc[0]) # 여러 사람이 있더라도 타겟 환자 1명만 추려내기 위한 고유 ID입니다.
        
        start_f = row['start_frame'] # 병합된 데이터프레임에서 현재 행의 시작 프레임 제한을 가져옵니다 (일반 영상은 NaN).
        end_f = row['end_frame'] # 병합된 데이터프레임에서 현재 행의 종료 프레임 제한을 가져옵니다 (일반 영상은 NaN).
        has_limit = pd.notna(start_f) and pd.notna(end_f) # 두 값이 모두 존재하여 프레임 구간 필터링이 필요한 상태인지 판별하는 플래그입니다.

        try:
            paths = path_list(common_path) # 사용자 정의 함수를 호출하여 필요한 하위 폴더들의 전체 절대 경로를 딕셔너리로 생성합니다.
        except Exception as e: # 경로 딕셔너리 생성 중 폴더명 오류 등으로 에러가 발생하면
            print(f"⚠️ 경로 생성 에러 ({common_path}): {e}") # 프로그램이 멈추지 않도록 원인만 출력하고
            continue # 해당 시퀀스는 건너뜁니다.

        json_dir = Path(paths['interp_data']) # 17 키포인트 보간이 완료된 JSON 폴더 경로를 Path 객체로 감쌉니다.
        txt_dir = Path(paths['yolo_txt']) # 변환된 결과물인 TXT 파일을 저장할 대상 폴더 경로입니다.
        img_dir = Path(paths['frame']) # 프레임 이미지가 들어있는 원본 폴더 경로입니다.

        if not json_dir.exists(): # JSON 폴더가 아예 만들어지지 않은 데이터라면
            continue # 변환할 라벨이 없으므로 시간 낭비 없이 건너뜁니다.
            
        txt_dir.mkdir(parents=True, exist_ok=True) # 저장할 TXT 폴더가 존재하지 않으면 자동으로 상위 폴더까지 묶어서 생성해 줍니다.

        # --- [💡 최적화 포인트: 이미지 크기 한 번만 읽기] ---
        cached_w, cached_h = None, None # 이 시퀀스 내내 재사용될 이미지의 너비와 높이 변수를 비워둔 채 준비합니다.
        
        first_img_path = img_dir / "000000.jpg" # 🌟 사용자 요청 로직: 가장 첫 번째 0번 프레임을 대표로 지정하여 해상도를 확인합니다.
        if not first_img_path.exists(): # 만약 000000.jpg 파일이 존재하지 않는다면 (데이터 누락 또는 다른 확장자)
            existing_imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) # 폴더 안의 모든 이미지를 긁어모아 리스트로 만듭니다.
            if not existing_imgs: continue # 긁어모았는데도 이미지가 한 장도 없다면 기준을 삼을 수 없어 패스합니다.
            first_img_path = existing_imgs[0] # 아쉬운 대로 폴더 내에 존재하는 가장 첫 번째 이미지를 대체제로 사용합니다.
            
        img = cv2.imread(str(first_img_path)) # 찾아낸 기준 이미지를 OpenCV로 메모리에 다차원 배열로 읽어옵니다.
        if img is None: continue # 파일이 손상되어 OpenCV가 이미지를 읽어들이지 못했다면 건너뜁니다.
        cached_h, cached_w = img.shape[:2] # 읽어온 이미지 배열에서 높이와 너비 정보만 쏙 빼내어 캐싱(저장)해 둡니다.
        # ---------------------------------------------------

        for json_path in json_dir.glob("*.json"): # json_dir 폴더 안의 모든 .json 파일을 하나씩 꺼내어 순회합니다.
            file_stem = json_path.stem # 'frame_0012.json' 등에서 확장자를 제외한 순수 파일명('frame_0012')만 가져옵니다.

            # --- [구간 필터링 로직] ---
            if has_limit: # 만약 데이터프레임에 시작/종료 프레임 제한이 명시된 데이터라면 (예: Bosanjin)
                stem_num_str = ''.join(filter(str.isdigit, file_stem)) # 파일명에서 텍스트를 걸러내고 순수하게 숫자만 합쳐 문자열로 만듭니다.
                if not stem_num_str: continue # 숫자가 아예 없다면 에러 방지를 위해 건너뜁니다.
                frame_num = int(stem_num_str) # 대소 비교를 위해 프레임 번호를 정수(int)로 변환합니다.
                
                if not (int(float(start_f)) <= frame_num <= int(float(end_f))): # 현재 파일의 번호가 지정된 구간을 벗어난다면
                    continue # 볼 필요가 없으므로 YOLO 변환을 수행하지 않고 다음 JSON 파일로 쿨하게 넘어갑니다.
            # --------------------------

            # 🌟 사용자 요청 로직: 변환된 결과물이 저장될 타겟 TXT 경로를 세팅합니다.
            out_path = txt_dir / f"{file_stem}.txt" # 원본 파일명과 완벽하게 동일한 이름의 텍스트 파일 저장 경로입니다.
            
            # 우리가 만들어둔 핵심 함수를 호출하여 단일 JSON의 타겟 인스턴스를 찾아 YOLO 라벨로 포맷팅합니다!
            is_success = convert_single_instance_to_yolo(
                json_path=json_path, # 읽어들일 소스 JSON 파일의 전체 경로를 넘깁니다.
                txt_path=out_path, # 변환 결과가 쓰일 타겟 TXT 파일 경로를 넘깁니다.
                img_w=cached_w, # 매 프레임마다 이미지를 여는 대신 한 번 저장해둔 너비를 재사용하여 성능을 극대화합니다.
                img_h=cached_h, # 캐싱해 둔 높이 값을 그대로 재사용합니다.
                patient_id=patient_id # 프레임 안의 여러 사람 중 타겟 환자 한 명만 골라내기 위한 기준 ID입니다.
            )
            
            if is_success: # 함수가 정상적으로 파일을 만들고 True를 반환했다면
                success_total += 1 # 총 성공 횟수 카운터를 1 증가시킵니다.

    print(f"\n🎉 [완료] 총 {success_total:,}개의 프레임 라벨이 빠르고 안전하게 YOLO 포맷으로 생성되었습니다!") # 모든 처리가 끝난 후 성공 개수를 쉼표와 함께 보기 좋게 출력합니다.