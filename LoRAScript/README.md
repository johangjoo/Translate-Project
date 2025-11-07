 Qwen 3 (8B) LoRA 번역 모델 학습


---

프로젝트 구조

```
project/
├── LoraScript/                    # 📦 모든 스크립트 파일
│   ├── setup.sh                   # 환경 설정
│   ├── requirements.txt           # 필수 패키지
│   ├── prepare_dataset.py         # 데이터 전처리
│   ├── train_qwen_lora.py         # 학습
│   └── README.md
               
├── training_data.jsonl            # 전처리된 전체 데이터
├── train.jsonl                    # 학습용
├── validation.jsonl               # 검증용 git에는 따로 포함 x
│
└── qwen3-8b-@@@@/      
    ├── lora_adapters/            
    └── qwen3-8b-@@@@/ # 최종 모델
```

---

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

--

## 🎓 학습 후 활용

### API 서버로 배포
- FastAPI로 REST API 구축
- vLLM으로 고속 추론
- Docker로 컨테이너화



**Happy Training! 🚀**
