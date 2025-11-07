"""
Windows용 Qwen LoRA 학습 스크립트 (진단 결과 기반 최적화)
- MAX_LENGTH: 512 → 160 (3배 빠름)
- 배치 크기 증가
- 예상 시간: 62시간 → 15-20시간
"""
import torch
import gc
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ==========================
# 설정 (진단 결과 기반 최적화)
# ==========================
# 🔥 핵심: MAX_LENGTH 대폭 축소 (512 → 160)
# 진단 결과: 95백분위 124토큰, 평균 88토큰
# 160이면 95% 커버 + 20% 여유
MAX_SEQ_LENGTH = 160

LOAD_IN_4BIT = True
SAMPLE_RATIO = 0.1

# LoRA 설정
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

# 모델 설정
MODEL_NAME = "Qwen/Qwen3-8b"
OUTPUT_DIR = str(PROJECT_ROOT / "qwen3-8b-lora-10ratio")
NEW_MODEL_NAME = "qwen3-8b-lora-10ratio"

TRAIN_FILE = str(PROJECT_ROOT / "train.jsonl")
VAL_FILE = str(PROJECT_ROOT / "validation.jsonl")

# ==========================
# 1. 4bit 양자화 설정
# ==========================
print("\n" + "="*60)
print("  🚀 진단 결과 기반 최적화")
print(f"  MAX_LENGTH: 512 → {MAX_SEQ_LENGTH} (3배 빠름)")
print(f"  데이터 샘플링: {SAMPLE_RATIO*100}%")
print(f"  목표 시간: 15-20시간")
print("="*60 + "\n")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

# ==========================
# 2. 모델 & Tokenizer 로드
# ==========================
print("🔄 모델 로딩 중...")
print(f"   모델: {MODEL_NAME}")
print(f"   4bit 양자화: {LOAD_IN_4BIT}\n")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ 모델 로드 완료\n")

# ==========================
# 3. LoRA 적용
# ==========================
print("🔧 LoRA 어댑터 추가 중...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\n학습 가능한 파라미터:")
model.print_trainable_parameters()
print()

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==========================
# 4. 데이터셋 로드
# ==========================
print("📚 데이터셋 로딩 중...")
# ==========================
# 4. 데이터셋 로드
# ==========================
print("📚 데이터셋 로딩 중...")

def formatting_prompts_func(examples):
    """
    Qwen3 messages 형식을 ChatML로 변환
    - enable_thinking=False (번역 태스크)
    """
    messages_list = examples["messages"]
    texts = []
    
    for messages in messages_list:
        # Qwen3의 apply_chat_template 사용
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False  # 번역 태스크에는 thinking 불필요
        )
        texts.append(text)
    
    return {"text": texts}

dataset = load_dataset(
    "json", 
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE
    },
    keep_in_memory=False
)

print(f"   원본 학습 데이터: {len(dataset['train']):,}개")
print(f"   원본 검증 데이터: {len(dataset['validation']):,}개")

# 샘플링
if SAMPLE_RATIO < 1.0:
    print(f"\n📊 {SAMPLE_RATIO*100}% 샘플링 중...")
    train_size = int(len(dataset['train']) * SAMPLE_RATIO)
    val_size = int(len(dataset['validation']) * SAMPLE_RATIO)
    
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_size))
    
    print(f"   샘플링 후 학습 데이터: {len(dataset['train']):,}개")
    print(f"   샘플링 후 검증 데이터: {len(dataset['validation']):,}개")

print()

# 포맷 적용
print("🔄 데이터 포맷 변환 중...")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    batch_size=500,
    remove_columns=dataset["train"].column_names,
    desc="Formatting prompts"
)

gc.collect()
print("✅ 데이터 준비 완료\n")

# ==========================
# 5. 학습 설정 (최적화)
# ==========================
print("⚙️  학습 설정 중 (최적화)...")

# 🔥 VRAM 여유(1.48GB)가 있으므로 배치 크기 증가 가능
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name=NEW_MODEL_NAME,
    
    # 🔥 배치 크기 증가 (VRAM 여유 활용)
    per_device_train_batch_size=6,      # 4 → 6
    gradient_accumulation_steps=3,       # 4 → 3
    per_device_eval_batch_size=4,        # 2 → 4
    
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    
    num_train_epochs=2,
    max_steps=-1,
    
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    logging_dir=f"{OUTPUT_DIR}/logs",
    
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    seed=42,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # 🔥 최적화 핵심
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,           # 160 (3배 빠름)
    packing=True,                        # 패킹 유지
    
    dataloader_num_workers=0,            # RAM 절약
    dataloader_pin_memory=False,
    
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ==========================
# 6. Trainer 초기화
# ==========================
print("🎯 Trainer 초기화 중...\n")

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

# ==========================
# 7. 예상 시간 출력
# ==========================
if torch.cuda.is_available():
    total_samples = len(dataset['train']) * sft_args.num_train_epochs
    effective_batch = sft_args.per_device_train_batch_size * sft_args.gradient_accumulation_steps
    total_steps = total_samples // effective_batch
    
    # 패킹 효과
    if sft_args.packing:
        total_steps = int(total_steps * 0.5)  # 진단에서 2배 효율 확인
    
    # 진단에서 측정한 스텝 시간: 25.3초 (MAX_LENGTH=512)
    # MAX_LENGTH를 160으로 줄이면: 512/160 = 3.2배 빠름
    estimated_step_time = 25.3 / 3.2  # 약 7.9초
    estimated_hours = total_steps * estimated_step_time / 3600
    
    print("="*60)
    print("🚀 학습 시작!")
    print("="*60)
    print(f"\n💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\n📊 최적화 내용:")
    print(f"   MAX_LENGTH: 512 → {MAX_SEQ_LENGTH} (3.2배 빠름)")
    print(f"   배치 크기: 4×4=16 → 6×3=18 (1.1배 빠름)")
    print(f"   총 개선: 약 3.5배 빠름")
    print(f"\n📊 학습 설정:")
    print(f"   데이터: {len(dataset['train']):,}개 ({SAMPLE_RATIO*100}%)")
    print(f"   에폭: {sft_args.num_train_epochs}")
    print(f"   실질 배치: {effective_batch}")
    print(f"   최대 길이: {MAX_SEQ_LENGTH}")
    print(f"   패킹: ✅ 활성화")
    print(f"\n⏱️  예상:")
    print(f"   기존 예상: 62.4시간")
    print(f"   개선 예상: ~{estimated_hours:.1f}시간")
    print(f"   저장 경로: {OUTPUT_DIR}\n")
    print("="*60 + "\n")

trainer_stats = trainer.train()

# ==========================
# 8. 모델 저장
# ==========================
print("\n" + "="*60)
print("💾 모델 저장 중...")
print("="*60 + "\n")

lora_path = f"{OUTPUT_DIR}/lora_adapters"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"✅ LoRA 어댑터 저장: {lora_path}")

print("\n🔄 LoRA 어댑터를 기본 모델에 병합 중...")
merged_model = model.merge_and_unload()
merged_path = f"{OUTPUT_DIR}/{NEW_MODEL_NAME}"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"✅ 병합 모델 저장: {merged_path}")

print("\n" + "="*60)
print("✅ 학습 완료!")
print("="*60)
print(f"\n📁 저장된 파일들:")
print(f"   1. LoRA 어댑터: {lora_path}")
print(f"   2. 병합 모델: {merged_path}")
print(f"   3. TensorBoard 로그: {OUTPUT_DIR}/logs")
print(f"\n📊 TensorBoard 실행:")
print(f"   tensorboard --logdir={OUTPUT_DIR}/logs")
print("\n" + "="*60)