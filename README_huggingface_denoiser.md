# Hugging Face 오디오 노이즈 제거 시스템

이 프로젝트는 Hugging Face의 SpeechBrain 모델을 사용하여 오디오에서 노이즈를 제거하는 시스템입니다.

## 🚀 주요 특징

- **사전 훈련된 모델**: SpeechBrain의 고성능 노이즈 제거 모델 사용
- **두 가지 모델 옵션**: 
  - `SpectralMaskEnhancement`: 일반적인 노이즈 제거 (16kHz)
  - `SepFormer`: 고급 음성 분리 및 향상 (8kHz)
- **GPU 지원**: CUDA 사용 가능 시 자동으로 GPU 가속
- **간편한 사용**: 단 몇 줄의 코드로 노이즈 제거 가능

## 📁 파일 구조

```
├── rnnaudio_huggingface.py      # 메인 Hugging Face 구현
├── test_huggingface_denoiser.py # 테스트 스크립트
├── requirements_huggingface.txt # 필수 라이브러리 목록
├── rnnaudio.py                  # 원본 RNN 구현 (참고용)
└── README_huggingface_denoiser.md # 이 파일
```

## 🛠️ 설치 방법

### 1. 필수 라이브러리 설치

```bash
pip install -r requirements_huggingface.txt
```

또는 개별 설치:

```bash
pip install speechbrain torch torchaudio librosa soundfile numpy
```

### 2. GPU 지원 (선택사항)

CUDA가 설치된 시스템에서 GPU 가속을 사용하려면:

```bash
# CUDA 11.8 예시
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 사용 방법

### 기본 사용법

```python
from rnnaudio_huggingface import HuggingFaceDenoiser, apply_denoising_huggingface

# 모델 초기화
denoiser = HuggingFaceDenoiser(model_type="spectral_mask", use_gpu=True)

# 노이즈 제거 실행
apply_denoising_huggingface("noisy_audio.wav", "clean_audio.wav", denoiser)
```

### 데모 실행

```python
from rnnaudio_huggingface import main
main()
```

또는 명령줄에서:

```bash
python rnnaudio_huggingface.py
```

## 🧪 테스트

시스템이 올바르게 설정되었는지 확인하려면:

```bash
python test_huggingface_denoiser.py
```

이 스크립트는 다음을 확인합니다:
- 필수 라이브러리 설치 상태
- GPU 사용 가능성
- 모델 로딩 가능성
- 데모 실행 가능성

## 📊 모델 비교

| 모델 | 샘플링 레이트 | 특징 | 권장 사용 |
|------|---------------|------|-----------|
| `spectral_mask` | 16kHz | 일반적인 노이즈 제거 | 대부분의 경우 |
| `sepformer` | 8kHz | 고급 음성 분리 | 복잡한 노이즈 환경 |

## 🔧 설정 옵션

### 모델 타입 변경

```python
# SpectralMaskEnhancement 사용 (기본값)
denoiser = HuggingFaceDenoiser(model_type="spectral_mask")

# SepFormer 사용
denoiser = HuggingFaceDenoiser(model_type="sepformer")
```

### GPU 사용 설정

```python
# GPU 사용 (CUDA 필요)
denoiser = HuggingFaceDenoiser(use_gpu=True)

# CPU 사용
denoiser = HuggingFaceDenoiser(use_gpu=False)
```

## 📝 지원 형식

- **입력**: WAV, FLAC, MP3 등 librosa가 지원하는 모든 형식
- **출력**: WAV 형식
- **권장 샘플링 레이트**: 
  - SpectralMask: 16kHz
  - SepFormer: 8kHz

## ⚠️ 주의사항

1. **첫 실행 시**: 모델이 자동으로 다운로드되므로 인터넷 연결과 충분한 디스크 공간이 필요합니다.
2. **메모리 사용량**: 큰 오디오 파일은 상당한 메모리를 사용할 수 있습니다.
3. **처리 시간**: GPU 사용 시 훨씬 빠른 처리가 가능합니다.

## 🐛 문제 해결

### 일반적인 문제들

1. **모듈 임포트 오류**
   ```bash
   pip install speechbrain
   ```

2. **CUDA 관련 오류**
   ```bash
   # CPU 모드로 실행
   denoiser = HuggingFaceDenoiser(use_gpu=False)
   ```

3. **메모리 부족**
   - 더 작은 오디오 파일로 테스트
   - GPU 메모리가 부족한 경우 CPU 모드 사용

## 📚 참고 자료

- [SpeechBrain 공식 문서](https://speechbrain.github.io/)
- [Hugging Face Audio Tasks](https://huggingface.co/tasks/audio-to-audio)
- [SepFormer 논문](https://arxiv.org/abs/2010.13154)

## 🤝 기여

버그 리포트나 개선 제안은 언제든 환영합니다!

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. SpeechBrain 모델의 라이선스를 확인하세요.
