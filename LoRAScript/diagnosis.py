"""
학습 속도 진단 스크립트
- GPU 사용률, 배치 크기, 스텝 시간 등 모든 정보 출력
- 이 정보로 병목 지점 파악 가능
"""
import torch
import time
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

# 설정
MAX_SEQ_LENGTH = 512
SAMPLE_RATIO = 0.10

MODEL_NAME = "Qwen/Qwen3-8b"
OUTPUT_DIR = str(PROJECT_ROOT / "qwen3-8b-translation-lora")
NEW_MODEL_NAME = "qwen3-8b-ko-ja-translation"

TRAIN_FILE = str(PROJECT_ROOT / "train.jsonl")
VAL_FILE = str(PROJECT_ROOT / "validation.jsonl")

print("\n" + "="*80)
print("  🔍 학습 속도 진단 스크립트")
print("="*80 + "\n")

# ==========================
# 1. 시스템 정보 출력
# ==========================
print("📊 시스템 정보:")
print("-" * 80)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU: {gpu_name}")
    print(f"   VRAM: {gpu_memory:.1f} GB")
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   PyTorch 버전: {torch.__version__}")
else:
    print("❌ GPU 없음 - CPU로 학습하면 엄청 느립니다!")

# CPU/RAM 정보
try:
    import psutil
    ram_total = psutil.virtual_memory().total / 1024**3
    cpu_count = psutil.cpu_count()
    print(f"\n💻 CPU: {cpu_count}개 코어")
    print(f"   RAM: {ram_total:.1f} GB")
except:
    print("\n💻 CPU/RAM 정보 확인 불가 (psutil 설치 필요)")

print()

# ==========================
# 2. 모델 로드 & 시간 측정
# ==========================
print("🔄 모델 로딩 중...")
start_time = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

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

load_time = time.time() - start_time
print(f"✅ 모델 로드 완료 ({load_time:.1f}초)\n")

# ==========================
# 3. LoRA 적용
# ==========================
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print()

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==========================
# 4. 데이터 로드 & 분석
# ==========================
print("📚 데이터셋 분석 중...")
start_time = time.time()

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        if input_text:
            text = f"""<|im_start|>system
You are a professional translator specializing in Korean-Japanese translation.<|im_end|>
<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        else:
            text = f"""<|im_start|>system
You are a professional translator specializing in Korean-Japanese translation.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
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
    train_size = int(len(dataset['train']) * SAMPLE_RATIO)
    val_size = int(len(dataset['validation']) * SAMPLE_RATIO)
    
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_size))
    
    print(f"\n   샘플링 후 학습 데이터: {len(dataset['train']):,}개")
    print(f"   샘플링 후 검증 데이터: {len(dataset['validation']):,}개")

# 데이터 길이 분석 (샘플 1000개)
print("\n🔍 데이터 길이 분석 중 (샘플 1000개)...")
sample_size = min(1000, len(dataset['train']))
lengths = []

for i in range(sample_size):
    example = dataset['train'][i]
    formatted = formatting_prompts_func({
        "instruction": [example["instruction"]],
        "input": [example["input"]],
        "output": [example["output"]]
    })
    tokens = tokenizer(formatted["text"][0], truncation=False)
    lengths.append(len(tokens['input_ids']))

import numpy as np
print(f"   평균 길이: {np.mean(lengths):.0f} 토큰")
print(f"   중앙값: {np.median(lengths):.0f} 토큰")
print(f"   최소: {np.min(lengths)} 토큰")
print(f"   최대: {np.max(lengths)} 토큰")
print(f"   95백분위: {np.percentile(lengths, 95):.0f} 토큰")
print(f"   99백분위: {np.percentile(lengths, 99):.0f} 토큰")

over_512 = sum(1 for l in lengths if l > 512)
print(f"\n   512 토큰 초과: {over_512}/{sample_size} ({over_512/sample_size*100:.1f}%)")
print(f"   현재 MAX_LENGTH: {MAX_SEQ_LENGTH}")

if np.percentile(lengths, 95) < MAX_SEQ_LENGTH * 0.7:
    recommended = int(np.percentile(lengths, 95) * 1.1)
    print(f"   ⚠️  추천 MAX_LENGTH: {recommended} (현재보다 짧게 설정 가능)")

print()

# 포맷 적용
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    batch_size=500,
    remove_columns=dataset["train"].column_names,
)

data_load_time = time.time() - start_time
print(f"✅ 데이터 준비 완료 ({data_load_time:.1f}초)\n")

gc.collect()

# ==========================
# 5. 학습 설정 진단
# ==========================
print("⚙️  학습 설정 분석:")
print("-" * 80)

# 여기서 실제 사용자 설정을 확인
PER_DEVICE_BATCH = 4  # 실제 코드의 값
GRAD_ACCUM = 4        # 실제 코드의 값
NUM_EPOCHS = 2

effective_batch = PER_DEVICE_BATCH * GRAD_ACCUM
total_samples = len(dataset['train']) * NUM_EPOCHS
total_steps_no_pack = total_samples // effective_batch

print(f"학습 데이터: {len(dataset['train']):,}개")
print(f"에폭 수: {NUM_EPOCHS}")
print(f"Per-device 배치: {PER_DEVICE_BATCH}")
print(f"Gradient accumulation: {GRAD_ACCUM}")
print(f"실질 배치 크기: {effective_batch}")
print(f"\n패킹 없을 때:")
print(f"   총 스텝: {total_steps_no_pack:,}")
print(f"   스텝당 3초 가정: {total_steps_no_pack * 3 / 3600:.1f}시간")

# 패킹 효과 추정
avg_tokens = np.mean(lengths)
packing_efficiency = min(MAX_SEQ_LENGTH / avg_tokens, 2.0)  # 최대 2배
total_steps_with_pack = int(total_steps_no_pack / packing_efficiency)

print(f"\n패킹 있을 때 (예상):")
print(f"   평균 길이: {avg_tokens:.0f} 토큰")
print(f"   패킹 효율: {packing_efficiency:.2f}배")
print(f"   예상 스텝: {total_steps_with_pack:,}")
print(f"   스텝당 3초 가정: {total_steps_with_pack * 3 / 3600:.1f}시간")

print()

# ==========================
# 6. 간단한 벤치마크 테스트
# ==========================
print("🔥 벤치마크 테스트 (3개 배치):")
print("-" * 80)
print("실제 학습 속도를 측정합니다...\n")

sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name=NEW_MODEL_NAME,
    
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    per_device_eval_batch_size=2,
    
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    
    num_train_epochs=1,
    max_steps=3,  # 3 스텝만 테스트
    
    eval_strategy="no",
    save_strategy="no",
    logging_steps=1,
    
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    report_to="none",
    
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
    packing=True,  # 패킹 상태 테스트
    
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

print("⏱️  3 스텝 테스트 중...")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024**3

benchmark_start = time.time()
trainer.train()
benchmark_time = time.time() - benchmark_start

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    current_mem = torch.cuda.memory_allocated() / 1024**3

avg_step_time = benchmark_time / 3

print(f"\n✅ 벤치마크 완료!")
print(f"   3 스텝 총 시간: {benchmark_time:.1f}초")
print(f"   평균 스텝 시간: {avg_step_time:.2f}초/스텝")

if torch.cuda.is_available():
    print(f"\n   GPU 메모리:")
    print(f"   - 현재 사용: {current_mem:.2f} GB")
    print(f"   - 최대 사용: {peak_mem:.2f} GB")
    print(f"   - 여유 공간: {gpu_memory - peak_mem:.2f} GB")

# ==========================
# 7. 최종 예상 시간 계산
# ==========================
print("\n" + "="*80)
print("📊 최종 예상 시간")
print("="*80)

print(f"\n패킹 없이 학습 시:")
print(f"   총 스텝: {total_steps_no_pack:,}")
print(f"   스텝당 시간: {avg_step_time:.2f}초")
print(f"   예상 총 시간: {total_steps_no_pack * avg_step_time / 3600:.1f}시간")

print(f"\n패킹 있게 학습 시 (현재 설정):")
print(f"   예상 스텝: {total_steps_with_pack:,}")
print(f"   스텝당 시간: {avg_step_time:.2f}초")
print(f"   예상 총 시간: {total_steps_with_pack * avg_step_time / 3600:.1f}시간")

# ==========================
# 8. 병목 지점 분석
# ==========================
print("\n" + "="*80)
print("🔍 병목 지점 분석")
print("="*80)

issues = []
recommendations = []

# GPU 체크
if not torch.cuda.is_available():
    issues.append("❌ CRITICAL: GPU가 없습니다! CPU로 학습하면 100배 이상 느립니다.")
    recommendations.append("→ GPU가 있는 환경에서 실행하세요.")
elif "GTX" in gpu_name or "MX" in gpu_name:
    issues.append(f"⚠️  GPU 성능 낮음: {gpu_name}")
    recommendations.append("→ RTX 3060 이상 권장")

# 배치 크기 체크
if PER_DEVICE_BATCH <= 2:
    issues.append(f"⚠️  배치 크기 작음: {PER_DEVICE_BATCH}")
    if torch.cuda.is_available() and (gpu_memory - peak_mem) > 4:
        recommendations.append(f"→ per_device_train_batch_size를 4-6으로 늘려보세요 (VRAM 여유: {gpu_memory - peak_mem:.1f}GB)")

# 시퀀스 길이 체크
if np.percentile(lengths, 95) < MAX_SEQ_LENGTH * 0.7:
    recommended_length = int(np.percentile(lengths, 95) * 1.1)
    issues.append(f"⚠️  MAX_LENGTH가 너무 큼: {MAX_SEQ_LENGTH} (95%가 {np.percentile(lengths, 95):.0f} 이하)")
    recommendations.append(f"→ MAX_SEQ_LENGTH를 {recommended_length}로 줄이면 더 빠름")

# 스텝 시간 체크
if avg_step_time > 5:
    issues.append(f"⚠️  스텝이 너무 느림: {avg_step_time:.2f}초/스텝")
    recommendations.append("→ GPU 사용률 확인 필요 (nvidia-smi 또는 작업 관리자)")
    
if avg_step_time < 1:
    issues.append(f"✅ 스텝 속도 매우 빠름: {avg_step_time:.2f}초/스텝")

# 워커 체크
if sft_args.dataloader_num_workers == 0:
    recommendations.append("→ dataloader_num_workers를 1-2로 시도해볼 수 있음 (RAM 모니터링 필요)")

print()
if not issues:
    print("✅ 큰 문제 없음!")
else:
    for issue in issues:
        print(issue)

print("\n💡 최적화 제안:")
if not recommendations:
    print("   현재 설정이 적절합니다.")
else:
    for rec in recommendations:
        print(rec)

print("\n" + "="*80)
print("📋 다음 단계:")
print("="*80)
print("\n1. 위 정보를 전부 복사해서 알려주세요")
print("2. 특히 다음 정보가 중요합니다:")
print("   - GPU 이름과 VRAM")
print("   - 평균 스텝 시간")
print("   - 예상 총 시간")
print("   - 병목 지점 분석 결과")
print("\n3. 현재 학습이 실제로 얼마나 걸리는지도 알려주세요")
print("   (20시간이라고 하셨는데, TensorBoard나 로그에서 확인한 값인가요?)")
print("\n" + "="*80)