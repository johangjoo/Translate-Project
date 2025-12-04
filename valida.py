import json
import random
import os
from pathlib import Path

# ================= 설정 =================
PROJECT_DIR = Path("C:/Works/Translate-Project") 
TRAIN_FILE = PROJECT_DIR / "train.jsonl"
NEW_VAL_FILE = PROJECT_DIR / "validation.jsonl"
VAL_SIZE_PER_DIR = 35000  
# =======================================

def get_direction(line_data):
    """메시지 내용으로 번역 방향 판단"""
    try:
        user_msg = line_data['messages'][1]['content']
        if '[Korean to Japanese]' in user_msg:
            return 'ko2ja'
        elif '[Japanese to Korean]' in user_msg:
            return 'ja2ko'
    except:
        return None
    return None

def main():
    print(f"📂 데이터 읽는 중: {TRAIN_FILE}")
    
    ko2ja_data = []
    ja2ko_data = []
    
    # 1. Train 파일 읽어서 분류
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                direction = get_direction(data)
                
                if direction == 'ko2ja':
                    ko2ja_data.append(line)
                elif direction == 'ja2ko':
                    ja2ko_data.append(line)
            except:
                continue

    print(f"📊 분석 결과:")
    print(f"   - 한→일 데이터: {len(ko2ja_data):,}개")
    print(f"   - 일→한 데이터: {len(ja2ko_data):,}개")
    print(f"   - 총 데이터: {len(ko2ja_data) + len(ja2ko_data):,}개")
    
    # 데이터 셔플
    random.seed(42)
    random.shuffle(ko2ja_data)
    random.shuffle(ja2ko_data)
    
    # 2. 검증 데이터 추출
    val_ko2ja = ko2ja_data[:VAL_SIZE_PER_DIR]
    val_ja2ko = ja2ko_data[:VAL_SIZE_PER_DIR]
    
    new_train_ko2ja = ko2ja_data[VAL_SIZE_PER_DIR:]
    new_train_ja2ko = ja2ko_data[VAL_SIZE_PER_DIR:]
    
    validation_set = val_ko2ja + val_ja2ko
    train_set = new_train_ko2ja + new_train_ja2ko
    
    random.shuffle(validation_set)
    random.shuffle(train_set)
    
    # 3. 파일 저장
    print(f"\n💾 저장 중...")
    
    # 기존 train 백업
    if os.path.exists(TRAIN_FILE):
        os.rename(TRAIN_FILE, str(TRAIN_FILE) + ".bak")
        print("   - 기존 train.jsonl 백업 완료 (.bak)")

    # 새 train 저장
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for line in train_set:
            f.write(line)
            
    # 새 validation 저장
    with open(NEW_VAL_FILE, 'w', encoding='utf-8') as f:
        for line in validation_set:
            f.write(line) # 이미 줄바꿈이 포함되어 있음

    print(f"\n✅ 완료되었습니다!")
    print(f"   - 새로운 Train: {len(train_set):,}개")
    print(f"   - 새로운 Validation: {len(validation_set):,}개 (한일/일한 각 {VAL_SIZE_PER_DIR}개)")
    print(f"   - 파일 위치: {NEW_VAL_FILE}")

if __name__ == "__main__":
    main()