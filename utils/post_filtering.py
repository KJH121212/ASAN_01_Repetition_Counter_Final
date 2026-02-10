from pathlib import Path 
from tqdm import tqdm
import json

def filter_and_save_json(src_kpt_dir, dst_kpt_dir, target_ids):
    src_path = Path(src_kpt_dir)
    dst_path = Path(dst_kpt_dir)
    
    # 출력 폴더 생성
    dst_path.mkdir(parents=True, exist_ok=True)
    
    if not src_path.exists():
        print(f"❌ 원본 경로가 없습니다: {src_path}")
        return False

    json_files = sorted(list(src_path.glob("*.json")))
    if not json_files:
        print("⚠️ 처리할 JSON 파일이 없습니다.")
        return False

    print(f"\nrunning Filtering... (Target IDs: {target_ids})")
    
    modified_count = 0
    
    for json_file in tqdm(json_files, desc="Filtering JSON"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 필터링 로직
            if 'instance_info' in data:
                filtered_instances = [
                    inst for inst in data['instance_info'] 
                    if inst.get('instance_id') in target_ids
                ]
                data['instance_info'] = filtered_instances
            
            # 새 위치에 저장
            save_path = dst_path / json_file.name
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            modified_count += 1

        except Exception as e:
            print(f"⚠️ Error processing {json_file.name}: {e}")
            
    print(f"✅ 필터링 완료: {modified_count}개의 파일이 '{dst_path.name}'에 저장됨.")
    return True