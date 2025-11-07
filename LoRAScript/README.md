# 🚀 Qwen 3 (8B) LoRA 번역 모델 학습

RTX 5070 Ti (12GB VRAM)에서 Qwen 3 (8B) 모델을 LoRA로 번역 작업에 특화시키는 완전한 가이드입니다.

---

## 📁 프로젝트 구조

```
your_project/
├── LoraScript/                    # 📦 모든 스크립트 파일
│   ├── setup.sh                   # 환경 설정
│   ├── requirements.txt           # 필수 패키지
│   ├── prepare_dataset.py         # 데이터 전처리
│   ├── train_qwen_lora.py         # 학습
│   ├── inference_lora.py          # 추론
│   ├── STEP1_환경설정.md
│   ├── STEP2_데이터준비.md
│   ├── STEP3_학습실행.md
│   └── README.md                  # 이 파일
│
├── json_data/                     # 📂 데이터 폴더
│   ├── ko_to_ja/                  # 한국어→일본어
│   │   ├── 801/
│   │   ├── 802/
│   │   └── ...
│   └── ja_to_ko/                  # 일본어→한국어
│       ├── 901/
│       └── ...
│
├── training_data.jsonl            # 전처리된 전체 데이터
├── train.jsonl                    # 학습용
├── validation.jsonl               # 검증용
│
└── qwen3-8b-translation-lora/      # 🎯 학습 결과
    ├── lora_adapters/             # LoRA 어댑터
    └── qwen3-8b-ko-ja-translation/ # 최종 모델
```

---

## ⚡ 빠른 시작 (4단계)

### 1️⃣ 환경 설정 (5분)

```bash
cd LoraScript
bash setup.sh
```

**상세 가이드:** [STEP1_환경설정.md](STEP1_환경설정.md)

---

### 2️⃣ 데이터 준비 (10분)

```bash
# 프로젝트 루트에서 숫자 폴더 이동
mv 801 802 803 json_data/ko_to_ja/
mv 901 902 json_data/ja_to_ko/

# LoraScript 폴더에서 전처리
cd LoraScript
python prepare_dataset.py
```

**상세 가이드:** [STEP2_데이터준비.md](STEP2_데이터준비.md)

---

### 3️⃣ 학습 실행 (1~10시간)

```bash
cd LoraScript
python train_qwen_lora.py
```

**상세 가이드:** [STEP3_학습실행.md](STEP3_학습실행.md)

---

### 4️⃣ 모델 테스트 (즉시)

```bash
cd LoraScript
python inference_lora.py
```

---

## 🎯 주요 특징

### ✨ 12GB VRAM 최적화
- **QLoRA 4bit 양자화** - 메모리 절약
- **Unsloth** - 2배 빠른 학습
- **Gradient Accumulation** - 실질적으로 큰 배치
- **Gradient Checkpointing** - 메모리 효율

### 🌐 양방향 번역
- **한국어 → 일본어**
- **일본어 → 한국어**
- 화자 정보 (성별, 연령) 반영

### 🔄 재귀 탐색
폴더만 옮기면 하위의 모든 JSON 파일을 자동으로 찾습니다!

```
json_data/ko_to_ja/
├── 801/
│   ├── file1.json         ← 찾음 ✅
│   └── subfolder/
│       └── file2.json     ← 이것도 찾음 ✅
└── 802/
    └── file3.json         ← 찾음 ✅
```

---

## 📊 성능 벤치마크

**RTX 5070 Ti (12GB VRAM) 기준:**

| 데이터 수 | 학습 시간 | VRAM 사용 | 최종 모델 크기 |
|----------|---------|----------|--------------|
| 5,000개  | 1~2시간  | ~10GB   | ~16GB       |
| 10,000개 | 2~3시간  | ~10GB   | ~16GB       |
| 20,000개 | 4~6시간  | ~10GB   | ~16GB       |
| 50,000개 | 8~12시간 | ~10GB   | ~16GB       |

---

## 🛠️ 고급 설정

### 메모리 부족 시

`train_qwen_lora.py` 수정:
```python
per_device_train_batch_size=1    # 배치 크기 줄이기
gradient_accumulation_steps=8     # 누적 스텝 늘리기
LORA_R = 8                        # LoRA rank 줄이기
MAX_SEQ_LENGTH = 1024             # 시퀀스 길이 줄이기
```

### 더 나은 품질

```python
num_train_epochs=5                # 에폭 늘리기
LORA_R = 32                       # LoRA rank 늘리기
learning_rate=1e-4                # 학습률 낮추기
```

---

## 📝 데이터 권장사항

| 데이터 양 | 용도 | 예상 품질 |
|----------|------|----------|
| 1,000~5,000 | 테스트 | 낮음 |
| 5,000~10,000 | 프로토타입 | 보통 |
| 20,000~30,000 | 실전 사용 | 좋음 |
| 50,000+ | 상용 서비스 | 매우 좋음 |

**권장: 최소 20,000개 이상**

---

## 🚨 문제 해결

### CUDA Out of Memory
→ [STEP3_학습실행.md](STEP3_학습실행.md#-문제-해결) 참고

### 데이터 관련 문제
→ [STEP2_데이터준비.md](STEP2_데이터준비.md#-문제-해결) 참고

### 환경 설정 문제
→ [STEP1_환경설정.md](STEP1_환경설정.md#-문제-해결) 참고

---

## 📖 각 단계 상세 가이드

1. **[STEP1_환경설정.md](STEP1_환경설정.md)** - Python, CUDA, Unsloth 설치
2. **[STEP2_데이터준비.md](STEP2_데이터준비.md)** - JSON 데이터 전처리
3. **[STEP3_학습실행.md](STEP3_학습실행.md)** - LoRA 학습 실행
4. **inference_lora.py** - 학습된 모델 테스트

---

## 💡 실전 팁

### 1. 점진적 학습
```bash
# 작은 데이터로 먼저 테스트
max_steps=100  # train_qwen_lora.py에서
```

### 2. 체크포인트 활용
학습 중단 시 자동으로 마지막 체크포인트에서 재개됩니다.

### 3. TensorBoard 모니터링
```bash
tensorboard --logdir=../qwen-8b-translation-lora/logs
```

---

## 🎓 학습 후 활용

### Python에서 사용
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../qwen3-8b-translation-lora/qwen3-8b-ko-ja-translation",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 번역 수행
prompt = "한국어를 일본어로 번역..."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### API 서버로 배포
- FastAPI로 REST API 구축
- vLLM으로 고속 추론
- Docker로 컨테이너화

---

## 📞 도움이 필요하면

1. **GPU 상태**: `nvidia-smi`
2. **학습 로그**: `../qwen3-8b-translation-lora/logs/`
3. **Python 버전**: `python --version` (3.10+ 필요)
4. **CUDA 버전**: `nvcc --version`

---

## 🎉 축하합니다!

이제 자신만의 한일 번역 모델을 학습했습니다!

**다음 단계:**
- 더 많은 데이터로 재학습
- 하이퍼파라미터 튜닝
- 실서비스 배포

---

**Happy Training! 🚀**
