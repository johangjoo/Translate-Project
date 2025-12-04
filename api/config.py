#모델 등 기본 사양 설정
import os
from pathlib import Path

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# 노이즈 제거 사용 여부
USE_DENOISER = True

# 업로드 설정
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE_MB = 100  # 최대 파일 크기 (MB)

# 서버 설정
HOST = "0.0.0.0"
PORT = 8000

# GPU 설정
USE_GPU = True

# 지원 오디오 포맷
SUPPORTED_AUDIO_FORMATS = [
    '.wav', '.mp3', '.m4a', '.flac', 
    '.ogg', '.aac', '.wma', '.opus'
]
# 모델 경로 설정
TRANSLATION_BASE_MODEL = BASE_DIR / "qwen3-14b-lora-10ratio"  # 기본값 (14B LoRA)
TRANSLATION_MODELS = {
    "qwen-8b-base": BASE_DIR / "models" / "qwen3-8b-base",  # 실제 폴더명에 맞춤
    "qwen-8b-lora": BASE_DIR / "qwen3-8b-lora-10ratio",
    "qwen-14b-base": BASE_DIR / "qwen3-14b-base",
    "qwen-14b-lora": BASE_DIR / "qwen3-14b-lora-10ratio",  # 기본값
}

# Whisper 모델 설정
WHISPER_MODEL_SIZE = "large-v3"  # tiny, base, small, medium, large, large-v3
DEFAULT_LANGUAGE = None  # None=자동감지, 또는 'ko', 'en', 'ja' 등

# 업로드 설정
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# 서버 설정
HOST = "0.0.0.0"
PORT = 8000