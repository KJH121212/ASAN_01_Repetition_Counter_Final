import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2 # 이미지 처리를 위한 OpenCV 라이브러리입니다.

# =================================================================
# 1. 설정 및 모듈 임포트
# =================================================================
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_Repetition_Counter_Final/")
sys.path.append(str(BASE_DIR))

from utils.path_list import path_list
from yolo.step1_dataset_builder import convert_single_instance_to_yolo

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "metadata_v2.1.csv"
BOSANJIN_PATH = DATA_DIR / "bosanjin_seg_data_v2.1.csv"

# =================================================================
# 2. 데이터 로드 및 전처리
# =================================================================
print(f"📂 CSV 로드 중... ({METADATA_PATH.name}, {BOSANJIN_PATH.name})")
meta_df = pd.read_csv(METADATA_PATH)
bosan_df = pd.read_csv(BOSANJIN_PATH)

# =================================================================
# 3. 데이터셋 순회 및 YOLO 포맷 변환 (최적화 버전)
# =================================================================
success_total = 0 # 전체 변환 성공 개수를 세기 위한 전역 카운터입니다.

for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="시퀀스 변환 중"): # DataFrame의 모든 행을 순회합니다.
    
    if row.get('is_train') == True or row.get('is_val') == True: # 학습용 또는 검증용 데이터만 필터링하여 작업합니다.
        
        common_path = row['common_path'] # 현재 시퀀스의 폴더 경로를 가져옵니다.
        patient_id = int(row['patient_id']) # 현재 시퀀스의 환자 ID를 정수형으로 가져옵니다.
        
        paths = path_list(common_path) # 해당 폴더의 상세 경로들을 딕셔너리로 불러옵니다.
        
        json_dir = paths['interp_data'] # JSON 파일들이 위치한 폴더 경로입니다.
        txt_dir = paths['yolo_txt'] # 변환된 TXT 파일이 저장될 폴더 경로입니다.
        img_dir = paths['frame'] # 원본 이미지 프레임들이 있는 폴더 경로입니다.
        
        if not json_dir.exists(): # JSON 폴더가 아예 존재하지 않는다면
            continue # 처리할 데이터가 없으므로 다음 시퀀스로 넘어갑니다.
            
        txt_dir.mkdir(parents=True, exist_ok=True) # 결과물을 저장할 폴더를 안전하게 생성합니다.
        
        # --- [💡 핵심 최적화 구간: 이미지 크기 한 번만 읽기] ---
        cached_w, cached_h = None, None # 이 폴더에서 사용할 이미지 너비와 높이를 초기화합니다.
        
        first_img_path = img_dir / "000000.jpg" # 첫 번째 기준이 될 이미지 경로를 설정합니다.
        if not first_img_path.exists(): # 만약 jpg 포맷이 없다면
            first_img_path = img_dir / "000000.png" # png 포맷으로 대체하여 찾습니다.
            
        # 000000 프레임이 없을 수도 있으니(이빨 빠진 데이터), 폴더 내 첫 번째 이미지를 안전하게 탐색합니다.
        if not first_img_path.exists(): 
            existing_imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) # 폴더 내 모든 이미지를 가져옵니다.
            if not existing_imgs: continue # 이미지가 단 한 장도 없다면 이 시퀀스는 패스합니다.
            first_img_path = existing_imgs[0] # 존재하는 가장 첫 번째 이미지를 기준으로 삼습니다.
            
        img = cv2.imread(str(first_img_path)) # 기준 이미지를 메모리로 읽어옵니다.
        if img is None: continue # 이미지가 손상되어 읽히지 않으면 패스합니다.
        
        cached_h, cached_w = img.shape[:2] # 높이와 너비를 추출하여 캐싱(저장)해 둡니다.
        # ----------------------------------------------------

        for json_path in json_dir.glob("*.json"): # 폴더 내 모든 JSON 파일을 하나씩 꺼냅니다.
            file_stem = json_path.stem # 확장자를 제외한 순수 파일명(예: "000002")을 추출합니다.
            
            out_path = txt_dir / f"{file_stem}.txt" # 저장될 결과물 TXT 파일의 절대 경로입니다.
            frame_path = img_dir / f"{file_stem}.jpg" # 쌍을 이룰 이미지 파일의 경로입니다.
            
            if not frame_path.exists(): # jpg가 존재하지 않는다면
                frame_path = img_dir / f"{file_stem}.png" # png로 재확인합니다.
                
            if not frame_path.exists(): # 매칭되는 이미지가 결국 없다면
                continue # 라벨만 존재할 수는 없으므로 변환을 건너뜁니다.
                
            # 매번 이미지를 열지 않고, 미리 구해둔 cached_w, cached_h를 재사용합니다!
            is_success = convert_single_instance_to_yolo(
                json_path=json_path,
                txt_path=out_path,
                img_w=cached_w, # 캐싱된 너비 사용
                img_h=cached_h, # 캐싱된 높이 사용
                patient_id=patient_id
            )
            
            if is_success: # 변환 및 저장이 무사히 완료되었다면
                success_total += 1 # 총 성공 횟수 카운터를 1 증가시킵니다.

print(f"\n🎉 [완료] 총 {success_total:,}개의 프레임이 빠르고 안전하게 YOLO 포맷으로 변환되었습니다!")