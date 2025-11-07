"""
Windows용 추론 스크립트
영어 프롬프트 버전
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 병합된 모델 경로
MODEL_PATH = str(PROJECT_ROOT / "qwen3-8b-translation-lora" / "qwen3-8b-ko-ja-translation")

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# ==========================
# 모델 로드
# ==========================
print("\n" + "="*60)
print("  Windows용 추론 - 학습된 모델 로딩")
print("="*60 + "\n")

print(f"🔄 모델 로딩 중: {MODEL_PATH}\n")

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("✅ 모델 로드 완료\n")

# ==========================
# 번역 함수 (영어 프롬프트)
# ==========================
def translate(text: str, direction: str = "ko->ja", 
              speaker_gender: str = "unknown", 
              speaker_age: str = "unknown"):
    """
    번역 수행 (영어 프롬프트)
    
    Args:
        text: 번역할 텍스트
        direction: "ko->ja" 또는 "ja->ko"
        speaker_gender: 화자 성별
        speaker_age: 화자 연령대
    """
    # 영어 프롬프트 (학습 시와 동일)
    if direction == "ko->ja":
        instruction = f"""Translate the following Korean to Japanese naturally.
Speaker: {speaker_gender}, {speaker_age}

Korean: {text}"""
    else:
        instruction = f"""Translate the following Japanese to Korean naturally.
Speaker: {speaker_gender}, {speaker_age}

Japanese: {text}"""
    
    prompt = f"""<|im_start|>system
You are a professional translator specializing in Korean-Japanese translation. Translate naturally while considering context, tone, and cultural nuances.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|im_start|>assistant\n" in result:
        translation = result.split("<|im_start|>assistant\n")[-1]
        translation = translation.split("<|im_end|>")[0].strip()
        return translation
    else:
        return result

# ==========================
# 테스트
# ==========================
if __name__ == "__main__":
    print("="*60)
    print("🌐 번역 모델 테스트 (영어 프롬프트)")
    print("="*60 + "\n")
    
    # 테스트 케이스
    test_cases = [
        {
            "text": "오늘 날씨가 정말 좋네요. 산책 가실래요?",
            "direction": "ko->ja",
            "gender": "여성",
            "age": "20대-30대"
        },
        {
            "text": "社長、長時間お待ちさせてしまい、申し訳ありません。",
            "direction": "ja->ko",
            "gender": "남성",
            "age": "30대-50대"
        },
        {
            "text": "이거 진짜 맛있다! 너도 먹어봐.",
            "direction": "ko->ja",
            "gender": "남성",
            "age": "10대-20대"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"[테스트 {i}]")
        print(f"원문: {test['text']}")
        print(f"방향: {test['direction']}")
        print(f"화자: {test['gender']}, {test['age']}")
        
        result = translate(
            test['text'],
            test['direction'],
            test['gender'],
            test['age']
        )
        
        print(f"번역: {result}")
        print("-" * 60 + "\n")
    
    # 대화형 모드
    print("\n💬 대화형 번역 모드 (종료: 'quit')\n")
    
    while True:
        text = input("번역할 텍스트: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        direction = input("방향 (ko->ja / ja->ko): ").strip() or "ko->ja"
        gender = input("화자 성별 (남성/여성): ").strip() or "unknown"
        age = input("화자 연령대: ").strip() or "unknown"
        
        result = translate(text, direction, gender, age)
        print(f"\n✅ 번역 결과: {result}\n")
        print("-" * 60 + "\n")