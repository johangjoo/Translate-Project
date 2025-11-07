"""
JSON 데이터를 LoRA 학습용 데이터셋으로 변환
폴더 구조 자유 버전 - json_data 폴더 전체를 재귀 탐색하여
origin_lang과 tl_trans_lang 필드로 자동 방향 판단
"""
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "json_data"  # json_data 폴더 전체 탐색
OUTPUT_FILE = PROJECT_ROOT / "training_data.jsonl"

def create_translation_prompt(data: Dict) -> Dict:
    """
    JSON 데이터에서 학습용 프롬프트 생성
    origin_lang과 tl_trans_lang으로 자동 방향 판단
    """
    # 화자 정보 추출
    gender = data.get("speaker_gender", "unknown")
    age_group = data.get("speaker_age_group", "unknown")
    
    # 원문과 번역문
    source_text = data.get("tc_text", "").strip()
    target_text = data.get("tl_trans_text", "").strip()
    
    # 방향 확인 (JSON 필드로 자동 판단!)
    origin_lang = data.get("origin_lang", "")
    trans_lang = data.get("tl_trans_lang", "")
    
    # 빈 데이터 스킵
    if not source_text or not target_text:
        return None
    
    # 영어 프롬프트 (방향 자동 판단)
    if "한국어" in origin_lang and "일본어" in trans_lang:
        # 한국어 → 일본어
        instruction = f"""Translate the following Korean to Japanese naturally.
Speaker: {gender}, {age_group}

Korean: {source_text}"""
        response = target_text
        
    elif "일본어" in origin_lang and "한국어" in trans_lang:
        # 일본어 → 한국어
        instruction = f"""Translate the following Japanese to Korean naturally.
Speaker: {gender}, {age_group}

Japanese: {source_text}"""
        response = target_text
    else:
        # 지원하지 않는 언어 쌍
        return None
    
    return {
        "instruction": instruction,
        "input": "",
        "output": response
    }

def process_all_json_files(input_dir: Path, output_file: Path):
    """
    json_data 폴더 전체를 재귀적으로 탐색하여 모든 JSON 파일 처리
    폴더 구조에 상관없이 JSON 파일만 찾아서 처리
    """
    if not input_dir.exists():
        print(f"❌ 폴더가 없습니다: {input_dir}")
        print(f"   프로젝트 구조: Translate-Project/json_data/...")
        return
    
    # 모든 JSON 파일 찾기 (재귀적으로)
    json_files = list(input_dir.rglob("*.json"))
    
    if len(json_files) == 0:
        print(f"❌ JSON 파일이 없습니다: {input_dir}")
        return
    
    print(f"📁 JSON 파일 발견: {len(json_files)}개")
    print(f"   위치: {input_dir}\n")
    
    training_samples = []
    skipped = 0
    direction_stats = {
        'ko_to_ja': 0,  # 한국어 → 일본어
        'ja_to_ko': 0,  # 일본어 → 한국어
        'unknown': 0     # 방향 미상
    }
    
    # 모든 JSON 파일 처리
    for json_file in tqdm(json_files, desc="JSON 처리 중"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 방향 판단을 위한 정보 추출
            origin_lang = data.get("origin_lang", "")
            trans_lang = data.get("tl_trans_lang", "")
            
            sample = create_translation_prompt(data)
            if sample:
                training_samples.append(sample)
                
                # 통계 업데이트
                if "한국어" in origin_lang and "일본어" in trans_lang:
                    direction_stats['ko_to_ja'] += 1
                elif "일본어" in origin_lang and "한국어" in trans_lang:
                    direction_stats['ja_to_ko'] += 1
                else:
                    direction_stats['unknown'] += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"⚠️  {json_file.name} 처리 실패: {e}")
            skipped += 1
    
    # JSONL 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print(f"✅ 전처리 완료!")
    print(f"{'='*60}")
    print(f"📊 방향별 통계:")
    print(f"   - 한국어→일본어: {direction_stats['ko_to_ja']}개")
    print(f"   - 일본어→한국어: {direction_stats['ja_to_ko']}개")
    print(f"   - 미상/기타: {direction_stats['unknown']}개")
    print(f"   - 총 성공: {len(training_samples)}개")
    print(f"   - 총 스킵: {skipped}개")
    print(f"   - 저장 위치: {output_file}")
    print(f"{'='*60}")

def split_train_val(input_file: Path, train_ratio: float = 0.95):
    """
    학습/검증 데이터 분리 (95:5)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    train_size = int(total * train_ratio)
    
    train_file = input_file.parent / "train.jsonl"
    val_file = input_file.parent / "validation.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:train_size])
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[train_size:])
    
    print(f"\n📊 데이터 분리 완료:")
    print(f"   - 학습: {train_size}개 → {train_file}")
    print(f"   - 검증: {total - train_size}개 → {val_file}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  JSON → 학습 데이터 변환 (폴더 구조 자유 버전)")
    print("="*60 + "\n")
    
    print("📂 폴더 구조:")
    print(f"   - 탐색 경로: {INPUT_DIR}")
    print(f"   - 존재 여부: {'✅' if INPUT_DIR.exists() else '❌'}")
    if INPUT_DIR.exists():
        # 하위 폴더 미리보기
        subdirs = [d.name for d in INPUT_DIR.iterdir() if d.is_dir()]
        if subdirs:
            print(f"   - 하위 폴더: {', '.join(subdirs[:5])}")
            if len(subdirs) > 5:
                print(f"     ... 외 {len(subdirs)-5}개")
    print()
    
    # 1. JSON → JSONL 변환
    process_all_json_files(INPUT_DIR, OUTPUT_FILE)
    
    # 2. 학습/검증 분리
    if OUTPUT_FILE.exists():
        print()
        split_train_val(OUTPUT_FILE)
