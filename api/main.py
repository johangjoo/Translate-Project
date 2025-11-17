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
    """서버 시작 시 안내 메시지만 출력 (모델은 요청 시 로딩)"""
    print("\n" + "="*70)
    print("🚀 Audio Translation API 서버 시작...")
    print("="*70 + "\n")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"📂 프로젝트 루트: {PROJECT_ROOT}\n")

    print("⚙️  모델은 이제 '요청이 들어올 때' 로드되고, 처리 후 가능한 한 언로드됩니다.")
    print()
    print("="*70)
    print("📡 서버 실행 중: http://127.0.0.1:8000")
    print("📚 API 문서: http://127.0.0.1:8000/docs")
    print()
    print("🎯 사용 가능한 기능:")

    print("   ✓ STT만             → /api/v1/transcribe (구현 시)")
    print("   ✓ 번역만             → /api/v1/translate-text")
    print("   ✓ 오디오 파이프라인  → /api/audio/process")
    print("   ✓ 상태 확인          → /api/v1/health")

    print("   ✓ 텍스트 번역         → /api/v1/translate-text")
    print("   ✓ 오디오 파이프라인   → /api/v1/audio/process")
    print("   ✓ 상태 확인           → /api/v1/health")

    print("   ✓ 텍스트 번역         → /api/v1/translate-text")
    print("   ✓ 오디오 파이프라인   → /api/v1/audio/process")
    print("   ✓ 상태 확인           → /api/v1/health")

    print("   ✓ 텍스트 번역         → /api/v1/translate-text")
    print("   ✓ 오디오 파이프라인   → /api/v1/audio/process")
    print("   ✓ 상태 확인           → /api/v1/health")

    print("   ✓ 텍스트 번역         → /api/v1/translate-text")
    print("   ✓ 오디오 파이프라인   → /api/v1/audio/process")
    print("   ✓ 상태 확인           → /api/v1/health")
    

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
            "translate": "/api/v1/translate-text",
            "audio_pipeline": "/api/v1/audio/process",
            "health": "/api/v1/health",
            "audio_health": "/api/v1/audio/health",
            "audio_memory": "/api/v1/audio/memory",
            "languages": "/api/v1/languages",
            "docs": "/docs"
        }
    }

 