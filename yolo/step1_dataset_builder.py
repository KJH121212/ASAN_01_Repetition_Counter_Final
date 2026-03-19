import os                   # 경로 및 심볼릭 링크 생성을 위한 모듈입니다.
import json                 # JSON 파일을 읽고 파싱하기 위한 모듈입니다.
import yaml                 # YOLO 설정 파일(data.yaml)을 생성하기 위한 모듈입니다.
import pandas as pd         # CSV 메타데이터를 쉽게 다루기 위한 라이브러리입니다.
from pathlib import Path    # 경로를 객체 지향적으로 안전하게 다루기 위한 내장 모듈입니다.
from tqdm import tqdm   # 콘솔 창에 진행률(Progress bar)을 예쁘게 보여주는 모듈입니다.

# ==========================================
# 1단계: 단일 JSON -> YOLO TXT 변환 (순수 함수)
# ==========================================
def convert_single_instance_to_yolo(json_path, txt_path, img_w, img_h, patient_id): # 환자 ID를 받아 해당 인스턴스만 찾아 YOLO 포맷으로 변환합니다.
    try:
        with open(json_path, 'r') as f:                                         # 변환할 JSON 파일을 읽기 모드로 엽니다.
            data = json.load(f)                                                 # JSON 파일의 내용을 파이썬 딕셔너리로 파싱합니다.
        
        if 'instance_info' not in data or not data['instance_info']:            # 객체 정보 리스트가 존재하는지 방어적으로 확인합니다.
            return False                                                        # 비어있다면 False를 반환하고 종료합니다.
            
        target_person = None                                                    # 우리가 변환할 타겟 객체를 담을 변수를 초기화합니다.
        
        for person in data['instance_info']:                                    # JSON 내의 모든 인식된 사람 객체를 하나씩 확인합니다.
            if person.get('instance_id') == patient_id:                         # 객체의 instance_id가 우리가 찾는 patient_id와 같은지 비교합니다.
                target_person = person                                          # 일치한다면 타겟 변수에 해당 객체 데이터를 저장합니다.
                break                                                           # 원하는 사람을 찾았으므로 더 이상 순회하지 않고 반복문을 탈출합니다.
                
        if target_person is None:                                               # 프레임 안에 해당 환자가 아예 안 찍혀있을 수도 있으므로 체크합니다.
            return False                                                        # 환자가 없다면 변환할 데이터가 없으므로 False를 반환합니다.
        
        # --- [BBox 추출 및 정규화] ---
        raw_bbox = target_person.get('bbox', [])                                # 찾은 target_person에서 BBox를 가져옵니다.
        if len(raw_bbox) < 4: return False                                      # BBox 데이터가 4개 미만이면 불완전하므로 중단합니다.

        x1, y1, x2, y2 = raw_bbox                                               # 리스트 값을 언패킹합니다.
        box_w = x2 - x1                                                         # 너비를 계산합니다.
        box_h = y2 - y1                                                         # 높이를 계산합니다.
        box_cx = x1 + (box_w / 2)                                               # 중심 X 좌표를 계산합니다.
        box_cy = y1 + (box_h / 2)                                               # 중심 Y 좌표를 계산합니다.

        yolo_bbox = [box_cx / img_w, box_cy / img_h, box_w / img_w, box_h / img_h] # 0~1 사이로 정규화합니다.

        # --- [💡 핵심 업데이트: 키포인트 추출 및 신뢰도 기반 가시성(v) 판별] ---
        raw_kpts = target_person.get('keypoints', [])                           # 키포인트 좌표 리스트를 가져옵니다.
        raw_scores = target_person.get('keypoint_scores', [])                   # 키포인트 신뢰도 점수 리스트를 함께 가져옵니다.
        selected_kpts = []                                                      # 결과물을 담을 빈 리스트를 생성합니다.
        
        threshold = 0.05                                                        # 가려짐(1)과 잘 보임(2)을 나누는 기준점입니다.

        for i in range(5, 17):                                                  # 5번부터 16번까지의 키포인트만 순회합니다.
            x, y = raw_kpts[i]                                                  # 원본 좌표를 추출합니다.
            
            # 안전장치: 혹시라도 scores 리스트가 없거나 길이가 짧을 경우를 대비합니다.
            score = raw_scores[i] if i < len(raw_scores) else 0.0               # 해당 관절의 점수를 추출합니다.
            
            if x > 0 and y > 0:                                                 # 좌표가 정상적으로 존재한다면
                v = 2 if score > threshold else 1                               # 점수가 0.07 초과면 2(보임), 0.07 이하면 1(가려짐)을 줍니다.
            else:                                                               # 좌표 자체가 없다면 (0 이하)
                v = 0                                                           # 0(무시)을 줍니다.
                
            selected_kpts.extend([x / img_w, y / img_h, v])                     # 정규화하여 리스트에 추가합니다.

        # --- [파일 저장] ---
        line = f"{patient_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)} {' '.join(f'{v:.6f}' for v in selected_kpts)}\n" # 환자 ID를 클래스 번호로 기록합니다.
        
        with open(txt_path, 'w') as f:                                          # 텍스트 파일을 쓰기 모드로 엽니다.
            f.write(line)                                                       # 문자열을 기록합니다.
            
        return True                                                             # 성공적으로 변환했음을 반환합니다.

    except Exception as e:                                                      # 예기치 못한 에러를 방어합니다.
        print(f"❌ Error in {Path(json_path).name}: {e}")                       # 에러 내용을 콘솔에 출력합니다.
        return False                                                            # 실패했으므로 False를 반환합니다.