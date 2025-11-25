"""
Windows용 Qwen3 LoRA 학습 스크립트 (TRL 0.23 - 올바른 방식)
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
from trl import SFTTrainer, SFTConfig  # ← SFTConfig 사용!

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 설정
MAX_SEQ_LENGTH = 160
LOAD_IN_4BIT = True
SAMPLE_RATIO = 0.2

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

MODEL_NAME = "Qwen/Qwen3-14B"
OUTPUT_DIR = str(PROJECT_ROOT / "qwen3-14b-lora-20ratio")
NEW_MODEL_NAME = "qwen3-14b-lora-20ratio"

TRAIN_FILE = str(PROJECT_ROOT / "train.jsonl")
VAL_FILE = str(PROJECT_ROOT / "validation.jsonl")

print("\n" + "="*70)
print("  🚀 Qwen3 LoRA 학습 - RTX 5070 Ti Notebook (12GB)")
print("="*70 + "\n")

# 1. 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

# 2. 모델 & Tokenizer 로드
print("🔄 모델 로딩 중...\n")

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

# 3. LoRA 적용
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

# 4. 데이터셋 로드
print("📚 데이터셋 로딩 중...")

def formatting_prompts_func(examples):
    """
    Qwen3 messages 형식을 ChatML로 변환
    - enable_thinking=False (번역 태스크)
    """
    messages_list = examples["messages"]
    texts = []
    
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
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

# 5. 학습 설정 (SFTConfig 사용!)
print("⚙️  학습 설정 중 (12GB VRAM 최적화)...\n")

sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name=NEW_MODEL_NAME,
    
    # 배치 크기
    per_device_train_batch_size=6,
    gradient_accumulation_steps=3,
    per_device_eval_batch_size=4,
    
    # 학습률
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    
    # 에폭
    num_train_epochs=2,
    max_steps=-1,
    
    # 평가 및 저장
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    logging_dir=f"{OUTPUT_DIR}/logs",
    
    # 정밀도
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    # 옵티마이저
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # 기타
    seed=42,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # 🔥 SFT 전용 파라미터
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,  # 160
    packing=True,
    
    # CPU/RAM 최적화
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    
    # 그래디언트 체크포인팅
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# 6. Trainer 초기화
print("🎯 Trainer 초기화 중...\n")

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

# 7. 학습 시작
print("="*70)
print("🚀 학습 시작!")
print("="*70)
print(f"\n💾 GPU: {torch.cuda.get_device_name(0)}")
print(f"   데이터: {len(dataset['train']):,}개")
print(f"   에폭: 2")
print(f"   MAX_LENGTH: {MAX_SEQ_LENGTH}")
print("="*70 + "\n")

trainer_stats = trainer.train()

# 8. 모델 저장
print("\n" + "="*70)
print("💾 모델 저장 중...")
print("="*70 + "\n")

lora_path = f"{OUTPUT_DIR}/lora_adapters"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"✅ LoRA 어댑터: {lora_path}")

merged_model = model.merge_and_unload()
merged_path = f"{OUTPUT_DIR}/{NEW_MODEL_NAME}"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"✅ 병합 모델: {merged_path}")

print("\n✅ 학습 완료!")
print(f"📊 TensorBoard: tensorboard --logdir={OUTPUT_DIR}/logs\n")