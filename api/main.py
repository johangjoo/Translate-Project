"""
FastAPI 메인 앱 - 통합 버전
- Whisper STT + Qwen3 번역
- 모든 기능을 하나의 서버에서!
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path

from api.routes import router
from api.inference import initialize_stt_models
from api.translation import initialize_translator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Audio Translation API",
    description="Whisper STT + Qwen3-8b Translation 통합 API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router, prefix="/api/v1", tags=["API"])


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모든 모델 로딩"""
    print("\n" + "="*70)
    print("🚀 Audio Translation API 서버 시작...")
    print("="*70 + "\n")
    
    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"📂 프로젝트 루트: {PROJECT_ROOT}\n")
    
    # 1. STT 모델 로딩
    print("🎤 [1/2] Whisper STT 모델 로딩...")
    initialize_stt_models(
        whisper_model_size="medium",  # tiny, base, small, medium, large
        language=None,  # 자동 감지
        use_denoiser=False  # 속도 우선
    )
    print()
    
    # 2. 번역 모델 로딩
    print("🌐 [2/2] Qwen3 번역 모델 로딩...")
    model_path = PROJECT_ROOT / "qwen3-8b-lora-10ratio/qwen3-8b-lora-10ratio"  # ✅ 맞음!  # ✅ 경로 중복 제거!
    
    if not model_path.exists():
        print(f"⚠️  경고: 모델 경로가 존재하지 않습니다: {model_path}")
        print(f"   상대 경로로 재시도...")
        model_path = "qwen3-8b-lora-10ratio"
    
    initialize_translator(
        model_path=str(model_path),
        use_gpu=True,
        load_in_4bit=True
    )
    print()
    
    print("="*70)
    print("✅ 모든 모델 로딩 완료!")
    print()
    print("📡 서버 실행 중: http://0.0.0.0:8000")
    print("📚 API 문서: http://0.0.0.0:8000/docs")
    print()
    print("🎯 사용 가능한 기능:")
    print("   ✓ STT만             → /api/v1/transcribe")
    print("   ✓ 번역만             → /api/v1/translate-text")
    print("   ✓ STT + 번역 (풀)    → /api/v1/audio-to-translation")
    print("   ✓ 상태 확인          → /api/v1/health")
    print("="*70 + "\n")


@app.get("/")
def root():
    """루트 엔드포인트"""
    return {
        "message": "🎤 → 📝 → 🌐 Audio Translation API",
        "version": "2.0.0",
        "description": "Whisper STT + Qwen3-8b Translation",
        "features": {
            "stt": "Whisper (99개 언어)",
            "translation": "Qwen3-8b LoRA (ko ↔ ja)",
            "pipeline": "음성 → 텍스트 → 번역"
        },
        "endpoints": {
            "transcribe": "/api/v1/transcribe",
            "translate": "/api/v1/translate-text",
            "full_pipeline": "/api/v1/audio-to-translation",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }