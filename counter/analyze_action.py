import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))  

from utils.extract_kpt import extract_id_keypoints, normalize_skeleton_array
from utils.path_list import path_list

def main():
    # 1. 메타데이터 불러오기
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    csv_path = DATA_DIR / "metadata_v2.1.csv"
    df = pd.read_csv(csv_path)
    
    biceps_df = df[df['common_path'].str.contains('biceps_curl', case=False, na=False)]
    print(f"🎯 분석 대상 Biceps Curl 영상 개수: {len(biceps_df)}개\n")

    # 🌟 2. 촬영 각도별 Y-좌표 거리(Shoulder Y - Wrist Y) 통계 저장소 초기화
    # flex(수축) = 어깨와 손목이 가까워짐 (최소 거리)
    # ext(이완) = 어깨와 손목이 멀어짐 (최대 거리)
    dist_stats = {
        'diagonal': {'flex': [], 'ext': []},
        'frontal': {'flex': [], 'ext': []},
        'lateral': {'flex': [], 'ext': []}
    }

    # 3. 데이터 순회 및 분석
    for idx, row in biceps_df.iterrows():
        common_path = str(row['common_path'])
        video_name = Path(common_path).name
        camera_angle = video_name.split('__')[0].lower()
        
        if camera_angle not in dist_stats: continue
            
        p_id = row['patient_id']
        if pd.isna(p_id): continue
        target_id = int(p_id)
        
        paths = path_list(common_path=common_path, create_dirs=False)
        json_dir = paths['keypoint']
        if not json_dir.exists(): continue
            
        raw_kpt = extract_id_keypoints(str(json_dir), target_id=target_id)
        if len(raw_kpt) == 0 or np.all(raw_kpt == 0): continue
            
        norm_kpt = normalize_skeleton_array(raw_kpt)
        
        l_dists, r_dists = [], []
        
        # 4. 각 프레임별 어깨-손목의 절대적 Y-좌표 거리 계산
        for frame in norm_kpt:
            if np.all(frame == 0): continue
                
            # 좌측: L-Shoulder(5), L-Wrist(9)
            if frame[5][2] != 0 and frame[9][2] != 0: 
                # 정규화된 배열에서 어깨와 손목의 Y좌표 차이 절대값
                l_dist = abs(frame[5][1] - frame[9][1])
                l_dists.append(l_dist)
                
            # 우측: R-Shoulder(6), R-Wrist(10)
            if frame[6][2] != 0 and frame[10][2] != 0: 
                r_dist = abs(frame[6][1] - frame[10][1])
                r_dists.append(r_dist)

        # 5. 각도별 딕셔너리에 5%(최대수축), 95%(최대이완) 백분위수 거리 저장
        if l_dists:
            dist_stats[camera_angle]['flex'].append(np.percentile(l_dists, 5))
            dist_stats[camera_angle]['ext'].append(np.percentile(l_dists, 95))
        if r_dists:
            dist_stats[camera_angle]['flex'].append(np.percentile(r_dists, 5))
            dist_stats[camera_angle]['ext'].append(np.percentile(r_dists, 95))

    # ==========================================
    # 6. Y-좌표 거리 기반 최종 Config 자동 생성
    # ==========================================
    print("\n" + "="*70)
    print("📊 Biceps Curl 정규화 Y-좌표 거리 기반 Config 자동 생성")
    print("="*70)
    
    print("EXERCISE_CONFIG['biceps_curl'] = {")
    
    for cam_angle, stats in dist_stats.items():
        if not stats['flex']:
            print(f"    # '{cam_angle}' 뷰에 대한 유효한 데이터가 없습니다.")
            continue
            
        avg_flex_dist = np.mean(stats['flex']) # 손목이 어깨와 가장 가까울 때 (수축)
        avg_ext_dist = np.mean(stats['ext'])   # 손목이 어깨와 가장 멀 때 (이완)
        
        # 안전 마진(Margin) 적용 (정규화된 값이므로 0.1~0.15 정도의 수치 마진을 줍니다)
        flex_threshold = avg_flex_dist + 0.15  # 수축: 이 거리 '보다 작아지면(<)' 
        ext_threshold = avg_ext_dist - 0.15    # 이완: 이 거리 '보다 커지면(>)'
        
        print(f"    # --- [실측 통계] {cam_angle.capitalize()} 뷰 (평균 수축 거리: {avg_flex_dist:.3f}, 평균 이완 거리: {avg_ext_dist:.3f}) ---")
        print(f"    '{cam_angle}': {{")
        
        # 🌟 계산 방식을 'angle'이 아닌 'y_distance'로 설정
        print("        'calc_method': 'y_distance',")
        # y_distance 방식의 경우 index[0]과 index[2]의 거리를 계산하게 되므로 7/8번(팔꿈치)은 사용하지 않지만 배열 형태를 유지합니다.
        print("        'joints': {'left': [5, 7, 9], 'right': [6, 8, 10]},")
        print("        'state_machine': {")
        print("            'start_state': 'relax',")
        print("            'active_state': 'flexion',")
        # 수축 완료 조건: 거리가 flex_threshold 보다 "작아질 때(<)"
        print(f"            'trigger_active': {{'threshold': {flex_threshold:.3f}, 'operator': '<', 'msg': '수축 완료! 천천히 내리세요.'}},")
        # 1회 완료 조건: 거리가 ext_threshold 보다 "커질 때(>)"
        print(f"            'trigger_start': {{'threshold': {ext_threshold:.3f}, 'operator': '>', 'msg': '1회 완료!'}}")
        print("        }")
        print("    },")
        
    print("}")

if __name__ == "__main__":
    main()