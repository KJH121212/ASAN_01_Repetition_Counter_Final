import os                   # 경로 및 심볼릭 링크 생성을 위한 모듈입니다.
import json                 # JSON 파일을 읽고 파싱하기 위한 모듈입니다.
import yaml                 # YOLO 설정 파일(data.yaml)을 생성하기 위한 모듈입니다.
import pandas as pd         # CSV 메타데이터를 쉽게 다루기 위한 라이브러리입니다.
import shutil # 파일 복사 등 고수준 파일 연산을 수행하기 위한 모듈을 불러옵니다.
from pathlib import Path    # 경로를 객체 지향적으로 안전하게 다루기 위한 내장 모듈입니다.
from tqdm import tqdm   # 콘솔 창에 진행률(Progress bar)을 예쁘게 보여주는 모듈입니다.


# ==========================================
# 1단계: 단일 JSON -> YOLO TXT 변환 (순수 함수)
# ==========================================
def convert_single_instance_to_yolo(json_path, txt_path, img_w, img_h, patient_id): # 환자 ID를 받아 해당 인스턴스만 찾아 YOLO 포맷으로 변환합니다.
    try:
        with open(json_path, 'r', encoding='utf-8') as f:                           # 변환할 JSON 파일을 읽기 모드로 안전하게 엽니다.
            data = json.load(f)                                                     # JSON 파일의 텍스트를 파이썬 딕셔너리로 파싱합니다.
        
        if 'instance_info' not in data or not data['instance_info']:                # 객체 정보 리스트가 존재하는지 방어적으로 확인합니다.
            return False                                                            # 데이터가 비어있다면 False를 반환하고 즉시 종료합니다.
            
        target_person = None                                                        # 우리가 추출할 타겟 객체를 담을 변수를 비워둔 채 초기화합니다.
        
        for person in data['instance_info']:                                        # JSON 내에 인식된 모든 사람 객체 리스트를 하나씩 확인합니다.
            if person.get('instance_id') == patient_id:                             # 객체의 ID가 함수 매개변수로 받은 타겟 환자 ID와 일치하는지 비교합니다.
                target_person = person                                              # 일치한다면 타겟 변수에 해당 객체 데이터를 통째로 저장합니다.
                break                                                               # 타겟을 찾았으므로 불필요한 추가 순회를 막기 위해 반복문을 탈출합니다.
                
        if target_person is None:                                                   # 프레임 안에 타겟 환자가 가려지거나 안 찍혔을 경우를 체크합니다.
            return False                                                            # 환자가 없다면 YOLO 파일로 만들 데이터가 없으므로 False를 반환합니다.
        
        # --- [BBox 추출 및 정규화] ---
        raw_bbox = target_person.get('bbox', [])                                    # 찾은 타겟 객체에서 4개의 좌표가 담긴 BBox 리스트를 가져옵니다.
        if len(raw_bbox) < 4: return False                                          # BBox 데이터가 4개 미만으로 깨져있다면 작업을 중단합니다.

        x1, y1, x2, y2 = raw_bbox                                                   # BBox 리스트의 픽셀 좌표값을 각각의 변수로 언패킹(Unpacking)합니다.
        box_w = x2 - x1                                                             # 우측 하단 x에서 좌측 상단 x를 빼서 바운딩 박스의 너비를 계산합니다.
        box_h = y2 - y1                                                             # 우측 하단 y에서 좌측 상단 y를 빼서 바운딩 박스의 높이를 계산합니다.
        box_cx = x1 + (box_w / 2)                                                   # 좌측 상단 x에 너비의 절반을 더해 중심점 X 좌표를 계산합니다.
        box_cy = y1 + (box_h / 2)                                                   # 좌측 상단 y에 높이의 절반을 더해 중심점 Y 좌표를 계산합니다.

        yolo_bbox = [box_cx / img_w, box_cy / img_h, box_w / img_w, box_h / img_h]  # YOLO 포맷에 맞게 모든 BBox 좌표를 0~1 사이의 비율로 정규화(Normalize)합니다.

        # --- [키포인트 추출 및 신뢰도 기반 가시성(v) 판별] ---
        raw_kpts = target_person.get('keypoints', [])                               # 타겟 객체의 17개 키포인트 좌표 리스트를 가져옵니다.
        raw_scores = target_person.get('keypoint_scores', [])                       # 각 키포인트가 얼마나 정확한지 나타내는 신뢰도 점수 리스트를 가져옵니다.
        selected_kpts = []                                                          # 최종 변환된 키포인트 데이터들을 담을 빈 리스트를 생성합니다.
        
        threshold = 0.05                                                            # 모델이 관절을 '가려짐(1)'과 '잘 보임(2)'으로 나눌 기준점입니다.

        for i in range(5, 17):                                                      # 얼굴 부위(0~4)를 제외한 몸통/팔다리 관절(5번~16번)만 순회합니다.
            x, y = raw_kpts[i]                                                      # 원본 픽셀 좌표를 언패킹합니다.
            
            # 안전장치: 점수(scores) 리스트가 누락되었거나 길이가 비정상적으로 짧은 경우 에러를 막습니다.
            score = raw_scores[i] if i < len(raw_scores) else 0.0                   # 인덱스가 안전하면 점수를 가져오고, 아니면 0.0을 부여합니다.
            
            if x > 0 and y > 0:                                                     # 화면 바깥이 아닌 정상적인 좌표에 관절이 찍혀있다면
                v = 2 if score > threshold else 1                                   # 점수가 기준치를 넘으면 2(보임), 낮으면 1(가려지거나 흐림)을 부여합니다.
            else:                                                                   # 좌표값이 0 이하(측정 불가)라면
                v = 0                                                               # 가시성 0(무시함)을 부여합니다.
                
            selected_kpts.extend([x / img_w, y / img_h, v])                         # 좌표를 0~1로 정규화하고 가시성 값과 함께 리스트에 이어 붙입니다.

        line = f"0 {' '.join(f'{v:.6f}' for v in yolo_bbox)} {' '.join(f'{v:.6f}' for v in selected_kpts)}\n" # 클래스 ID 0번과 BBox, 키포인트 데이터를 공백으로 이어붙입니다.
        
        with open(txt_path, 'w', encoding='utf-8') as f:                            # 텍스트 파일을 쓰기 모드로 안전하게 엽니다.
            f.write(line)                                                           # 한 줄로 예쁘게 조립된 문자열을 파일에 기록합니다.
            
        return True                                                                 # 모든 과정을 무사히 통과했으므로 성공을 의미하는 True를 반환합니다.

    except Exception as e:                                                          # 파일 읽기/쓰기 중 발생할 수 있는 치명적인 에러를 잡아냅니다.
        print(f"❌ Error in {Path(json_path).name}: {e}")                           # 스크립트가 멈추지 않도록 콘솔에 에러 원인만 출력합니다.
        return False

def create_yolo_dataset_structure(df, dataset_dir, data_dir, step=1):
    """
    데이터프레임의 정보를 바탕으로 YOLO 모델 학습에 필요한 폴더 구조를 만들고,
    이미지와 라벨 파일을 지정된 간격(step)으로 추출하여 배치하는 함수입니다.
    """
    print(f"🚀 데이터셋 구조화 시작 (샘플링 간격: {step})") # 사용자에게 작업 시작과 적용된 간격을 알립니다.
    
    # 1. YOLO 폴더 구조 생성
    for split in ['train', 'val']:                                  # 학습용(train)과 검증용(val) 폴더를 각각 생성하기 위해 반복합니다.
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True) # 이미지 폴더를 안전하게 생성합니다 (이미 존재하면 건너뜁니다).
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True) # 라벨 폴더를 안전하게 생성합니다.

    counts = {'train': 0, 'val': 0, 'skip': 0, 'fixed': 0} # 처리 결과를 추적하고 요약하기 위한 카운터 딕셔너리를 초기화합니다.

    # 2. 데이터프레임 순회 및 파일 처리
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="파일 링킹 작업 중"): # 데이터프레임의 각 행을 순회하며 진행률 바를 표시합니다.
        if row.get('is_train') == True: split = 'train'     # 데이터가 훈련용으로 마킹되었다면 타겟 폴더를 'train'으로 지정합니다.
        elif row.get('is_val') == True: split = 'val'       # 데이터가 검증용으로 마킹되었다면 타겟 폴더를 'val'로 지정합니다.
        else: continue                                      # 둘 다 아니라면 불필요한 데이터이므로 처리를 건너뜁니다.

        common_path = row['common_path'] # 원본 파일들이 위치한 기준 경로 문자열을 가져옵니다.
        src_label_dir = data_dir / "5_YOLO_TXT" / common_path   # 변환 완료된 원본 라벨(txt) 폴더 경로를 생성합니다.
        src_image_dir = data_dir / "1_FRAME" / common_path      # 원본 프레임(image) 폴더 경로를 생성합니다.

        if not src_label_dir.exists() or not src_image_dir.exists():    # 라벨이나 이미지 폴더 중 하나라도 없으면 에러가 나므로 검사합니다.
            continue                                                    # 없는 경우 다음 시퀀스로 안전하게 넘어갑니다.

        label_files = sorted(list(src_label_dir.glob("*.txt")))     # 폴더 내의 모든 txt 라벨 파일을 이름순으로 정렬하여 리스트에 담습니다.
        if not label_files: continue                                # 라벨 파일이 하나도 비어있다면 에러를 방지하기 위해 건너뜁니다.

        # 3. 구간(Start/End) 필터링 적용
        start_f = row.get('start_frame')    # 메타데이터에서 특정 구간의 시작 프레임 번호를 가져옵니다.
        end_f = row.get('end_frame')        # 메타데이터에서 특정 구간의 종료 프레임 번호를 가져옵니다.

        if pd.notna(start_f) and pd.notna(end_f):   # 두 값이 모두 존재(결측치가 아님)하는 세그먼트 데이터인 경우 구간 필터링을 수행합니다.
            filtered_labels = []                        # 조건에 맞는 라벨만 임시로 담아둘 리스트입니다.
            for lf in label_files:                      # 정렬된 모든 라벨 파일을 순회합니다.
                stem_num_str = ''.join(filter(str.isdigit, lf.stem)) # 파일명에서 텍스트를 무시하고 숫자만 추출하여 문자열로 만듭니다 (예: frame_015 -> 015).
                if stem_num_str: # 숫자가 정상적으로 추출되었다면
                    frame_num = int(stem_num_str) # 대소 비교를 위해 추출된 문자열을 정수형(int)으로 변환합니다.
                    if int(start_f) <= frame_num <= int(end_f): # 현재 파일의 프레임 번호가 설정된 시작~종료 구간 안에 포함되는지 엄격하게 검사합니다.
                        filtered_labels.append(lf) # 조건에 맞으면 필터링 리스트에 추가합니다.
            label_files = filtered_labels # 전체 리스트를 우리가 원하는 구간 내의 파일들로만 완전히 교체합니다.

        # 4. 샘플링(Step) 간격 적용
        sampled_files = label_files[::step] # 필터링된 파일 리스트 중 step 간격마다 1개씩만 추출하여 학습 효율을 높입니다.

        # 5. 심볼릭 링크 및 복사 작업
        for label_file in sampled_files: # 샘플링까지 완료된 파일들을 하나씩 최종 타겟 폴더로 보냅니다.
            file_stem = label_file.stem # 파일의 확장자를 제외한 순수 이름만 가져옵니다.
            
            image_file = src_image_dir / f"{file_stem}.jpg" # 해당 라벨과 매칭되는 jpg 이미지 경로를 구성합니다.
            if not image_file.exists(): # 만약 jpg 파일이 존재하지 않는다면
                image_file = src_image_dir / f"{file_stem}.png" # png 확장자일 수도 있으므로 대체하여 다시 찾습니다.
            if not image_file.exists(): continue # 이미지가 아예 없다면 짝이 맞지 않으므로 이 프레임은 건너뜁니다.

            safe_common_path = common_path.replace("/", "_").replace("\\", "_") # 파일 덮어쓰기 충돌을 막기 위해 경로의 슬래시를 언더바로 치환합니다.
            unique_name = f"{safe_common_path}_{file_stem}" # 폴더 경로와 프레임 번호를 결합하여 전역적으로 고유한 파일명을 생성합니다.

            dst_image = dataset_dir / 'images' / split / f"{unique_name}{image_file.suffix}" # 최종적으로 링크가 생성될 타겟 이미지 경로입니다.
            dst_label = dataset_dir / 'labels' / split / f"{unique_name}.txt" # 최종적으로 복사될 타겟 라벨 경로입니다.

            if dst_image.is_symlink() and not dst_image.exists(): # 이전에 스크립트를 돌리다가 깨진 심볼릭 링크가 남아있는지 검사합니다.
                dst_image.unlink() # 연결이 끊긴 깨진 링크 파일은 시스템 오류 방지를 위해 삭제합니다.

            if dst_image.exists() and dst_label.exists(): # 이미지와 라벨이 이미 타겟 폴더에 완벽히 존재한다면
                counts['skip'] += 1 # 스킵 카운트를 올리고
                continue # 중복 작업을 건너뛰어 스크립트 실행 속도를 극대화합니다.

            try:
                if not dst_image.exists(): # 타겟 목적지에 이미지가 없다면
                    os.symlink(image_file, dst_image) # 무거운 이미지 파일을 직접 복사하지 않고 원본을 가리키는 심볼릭 링크(바로가기)를 생성하여 디스크 용량을 획기적으로 절약합니다.
                    counts['fixed'] += 1 # 새로 생성한 이미지 링크 카운트를 올립니다.
                
                if not dst_label.exists(): # 타겟 목적지에 라벨 파일이 없다면
                    shutil.copy2(label_file, dst_label) # 라벨(txt)은 용량이 매우 작으므로 원본의 메타데이터를 유지한 채 직접 안전하게 복사합니다.
                
                counts[split] += 1 # 이 프레임이 성공적으로 할당된 훈련(train) 또는 검증(val) 데이터 카운트를 1 증가시킵니다.
                
            except OSError as e: # 윈도우 환경 권한 문제 등으로 파일 시스템 에러가 발생하면
                print(f"⚠️ 파일 처리 에러 ({unique_name}): {e}") # 스크립트가 튕기지 않도록 에러 메시지만 출력하고 계속 진행합니다.

    # 6. data.yaml 파일 생성
    yaml_content = { # YOLO 모델이 학습을 시작할 때 반드시 읽어야 하는 데이터셋 설정 딕셔너리를 구성합니다.
        'path': str(dataset_dir.absolute()), # 데이터셋 최상단 폴더의 절대 경로를 문자열로 지정하여 모델이 파일 위치를 헤매지 않도록 합니다.
        'sampling_step': step, # 나중에 실험을 재현할 수 있도록 샘플링에 사용된 간격(step)을 메타 정보로 기록합니다.
        'train': 'images/train', # 모델이 훈련용 이미지들을 찾을 상대 경로입니다.
        'val': 'images/val', # 모델이 검증용 이미지들을 찾을 상대 경로입니다.
        'names': {0: 'person'}, # 우리가 탐지할 클래스 ID 0번의 이름을 'person'으로 보기 좋게 매핑합니다.
        'kpt_shape': [12, 3], # 모델이 예측할 키포인트의 차원(12개의 관절 포인트, x/y/가시성 3차원)을 명시합니다.
        'flip_idx': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10] # 데이터를 증강(Augmentation)할 때, 이미지를 좌우 반전시키면 대응되는 관절(왼팔<->오른팔)도 스왑되도록 인덱스 규칙을 설정합니다.
    }

    yaml_path = dataset_dir / "data.yaml" # 완성된 설정을 저장할 yaml 파일의 절대 경로 객체입니다.
    with open(yaml_path, 'w', encoding='utf-8') as f: # 문자가 깨지지 않도록 utf-8 인코딩 쓰기 모드로 파일을 안전하게 엽니다.
        yaml.dump(yaml_content, f, sort_keys=False) # 파이썬 딕셔너리를 YAML 형식으로 예쁘게 변환하여 파일에 저장합니다 (키 순서는 우리가 작성한 그대로 유지).

    print(f"\n📊 작업 요약:") # 모든 루프가 끝난 후 사용자에게 작업 결과를 보기 좋게 출력합니다.
    print(f"   - 생성된 Train 이미지: {counts['train']:,} 장") # 훈련용으로 최종 준비된 데이터 개수를 콤마(,)를 포함해 가독성 좋게 출력합니다.
    print(f"   - 생성된 Val 이미지:   {counts['val']:,} 장") # 검증용으로 준비된 데이터 개수를 출력합니다.
    
    return yaml_path # 메인 스크립트의 generated_yaml 변수로 할당될 수 있도록, 생성된 yaml 파일의 경로를 최종적으로 반환합니다.
