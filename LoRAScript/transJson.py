import json
import os
import shutil
from tqdm import tqdm
from datetime import datetime

def transform_line(line):
    """instruction/input/output을 태그 포함 Qwen3 messages 형식으로 변환"""
    try:
        data = json.loads(line.strip())
        instruction = data.get('instruction', '')
        input_text = data.get('input', '')
        output_text = data.get('output', '')
        
        # 번역 방향 파악 및 태그 설정
        if 'Korean to Japanese' in instruction or 'Korean->Japanese' in instruction:
            tag = "[Korean to Japanese]"
            direction = "ko2ja"
        elif 'Japanese to Korean' in instruction or 'Japanese->Korean' in instruction:
            tag = "[Japanese to Korean]"
            direction = "ja2ko"
        else:
            # 방향을 알 수 없는 경우 - 데이터 확인 필요
            print(f"⚠️  경고: 방향 파악 불가 - {instruction[:50]}")
            return None
        
        # Qwen3 messages 형식으로 변환
        return json.dumps({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional Korean-Japanese bilingual translator."
                },
                {
                    "role": "user",
                    "content": f"{tag}\n{input_text}"
                },
                {
                    "role": "assistant",
                    "content": output_text
                }
            ]
        }, ensure_ascii=False), direction
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return None

def transform_file(input_file):
    """파일 변환"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{input_file}.backup_{timestamp}"
    
    print(f"📦 백업 생성: {backup_file}")
    shutil.copy2(input_file, backup_file)
    
    temp_output = f"{input_file}.temp"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"📊 총 {total_lines:,}개 라인 변환 중...")
    
    ko_to_ja_count = 0
    ja_to_ko_count = 0
    fail_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(temp_output, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="변환"):
            result = transform_line(line)
            
            if result:
                json_str, direction = result
                outfile.write(json_str + '\n')
                
                if direction == "ko2ja":
                    ko_to_ja_count += 1
                elif direction == "ja2ko":
                    ja_to_ko_count += 1
            else:
                fail_count += 1
    
    os.replace(temp_output, input_file)
    
    success_count = ko_to_ja_count + ja_to_ko_count
    print(f"✅ 성공: {success_count:,}개")
    print(f"   ├─ 한국어→일본어: {ko_to_ja_count:,}개")
    print(f"   └─ 일본어→한국어: {ja_to_ko_count:,}개")
    print(f"❌ 실패: {fail_count:,}개\n")
    
    return success_count, fail_count, ko_to_ja_count, ja_to_ko_count

def show_samples(file_path, num_samples=2):
    """변환 결과 샘플 출력 (양방향 각각)"""
    print(f"📄 {os.path.basename(file_path)} 샘플:")
    
    ko_to_ja_samples = []
    ja_to_ko_samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_content = data['messages'][1]['content']
            
            if '[Korean to Japanese]' in user_content:
                ko_to_ja_samples.append(data)
            elif '[Japanese to Korean]' in user_content:
                ja_to_ko_samples.append(data)
            
            if len(ko_to_ja_samples) >= num_samples and len(ja_to_ko_samples) >= num_samples:
                break
    
    # 한→일 샘플
    if ko_to_ja_samples:
        print(f"\n  💬 한국어→일본어 샘플:")
        for i, data in enumerate(ko_to_ja_samples[:num_samples], 1):
            system_msg = data['messages'][0]['content']
            user_msg = data['messages'][1]['content']
            assistant_msg = data['messages'][2]['content']
            
            # 태그 이후 텍스트만 추출
            input_text = user_msg.split('\n', 1)[1][:50] if '\n' in user_msg else user_msg[:50]
            output_text = assistant_msg[:50]
            
            print(f"    [{i}]")
            print(f"        System: {system_msg}")
            print(f"        User: [Korean to Japanese]")
            print(f"              {input_text}{'...' if len(input_text) == 50 else ''}")
            print(f"        Assistant: {output_text}{'...' if len(output_text) == 50 else ''}")
    
    # 일→한 샘플
    if ja_to_ko_samples:
        print(f"\n  💬 일본어→한국어 샘플:")
        for i, data in enumerate(ja_to_ko_samples[:num_samples], 1):
            system_msg = data['messages'][0]['content']
            user_msg = data['messages'][1]['content']
            assistant_msg = data['messages'][2]['content']
            
            # 태그 이후 텍스트만 추출
            input_text = user_msg.split('\n', 1)[1][:50] if '\n' in user_msg else user_msg[:50]
            output_text = assistant_msg[:50]
            
            print(f"    [{i}]")
            print(f"        System: {system_msg}")
            print(f"        User: [Japanese to Korean]")
            print(f"              {input_text}{'...' if len(input_text) == 50 else ''}")
            print(f"        Assistant: {output_text}{'...' if len(output_text) == 50 else ''}")
    
    print()

def main():
    """메인 함수"""
    # ============================================
    # 🔧 여기만 수정하세요!
    # ============================================
    project_folder = "C:/Works/Translate-Project"  # 프로젝트 폴더명
    
    files_to_transform = [
        "train.jsonl",
        "training_data.jsonl", 
        "validation.jsonl"
    ]
    # ============================================
    
    print("=" * 70)
    print("🚀 Qwen3 Messages 형식 변환 스크립트 (태그 포함)")
    print("=" * 70)
    print(f"📁 작업 폴더: {project_folder}")
    print(f"📋 변환 형식: instruction/input/output → messages (with tags)")
    print(f"🏷️  태그: [Korean to Japanese], [Japanese to Korean]")
    print(f"💬 System: You are a professional Korean-Japanese bilingual translator.\n")
    
    if not os.path.exists(project_folder):
        print(f"❌ 오류: '{project_folder}' 폴더를 찾을 수 없습니다!")
        return
    
    total_success = 0
    total_fail = 0
    total_ko_to_ja = 0
    total_ja_to_ko = 0
    
    for filename in files_to_transform:
        file_path = os.path.join(project_folder, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️  건너뛰기: {filename} 파일이 없습니다.\n")
            continue
        
        print(f"\n{'='*70}")
        print(f"🔄 처리 중: {filename}")
        print(f"{'='*70}")
        
        success, fail, ko_to_ja, ja_to_ko = transform_file(file_path)
        total_success += success
        total_fail += fail
        total_ko_to_ja += ko_to_ja
        total_ja_to_ko += ja_to_ko
        
        # 샘플 출력
        show_samples(file_path)
    
    # 최종 결과
    print("=" * 70)
    print("🎉 모든 변환 완료!")
    print("=" * 70)
    print(f"✅ 총 성공: {total_success:,}개")
    print(f"   ├─ 한국어→일본어: {total_ko_to_ja:,}개")
    print(f"   └─ 일본어→한국어: {total_ja_to_ko:,}개")
    print(f"❌ 총 실패: {total_fail:,}개")
    print(f"\n💡 원본 파일들은 .backup_YYYYMMDD_HHMMSS 형식으로 백업되었습니다.")
    print("=" * 70)

if __name__ == "__main__":
    main()