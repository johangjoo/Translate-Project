"""
API 라우트 - 통합 버전
- STT API (기존)
- 번역 API (새로 추가)
- 전체 파이프라인 (새로 추가)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from pathlib import Path
import os
import uuid
import shutil
import time
from typing import Optional

router = APIRouter()

# 업로드 디렉토리
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ===== Pydantic 모델 =====

class STTResponse(BaseModel):
    """STT 응답"""
    text: str = Field(..., description="변환된 텍스트")
    language: str = Field(..., description="감지된 언어")
    audio_filename: str = Field(..., description="원본 파일명")
    processing_time: float = Field(..., description="처리 시간(초)")
    segments: Optional[list] = Field(None, description="세그먼트 정보")


class TranslationResponse(BaseModel):
    """번역 응답"""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    processing_time: float


class PipelineResponse(BaseModel):
    """전체 파이프라인 응답"""
    original_audio: str = Field(..., description="원본 오디오 파일명")
    transcribed_text: str = Field(..., description="STT 결과")
    translated_text: str = Field(..., description="번역 결과")
    detected_language: str = Field(..., description="감지된 언어")
    target_language: str = Field(..., description="목표 언어")
    processing_time: float = Field(..., description="총 처리 시간")
    stt_time: float = Field(..., description="STT 시간")
    translation_time: float = Field(..., description="번역 시간")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    stt_loaded: bool
    translator_loaded: bool
    stt_device: str
    translator_device: str


# ===== 유틸리티 함수 =====

def save_upload_file(upload_file: UploadFile, max_size_mb: int = 200) -> str:
    """업로드 파일 저장"""
    upload_file.file.seek(0, 2)
    file_size = upload_file.file.tell()
    upload_file.file.seek(0)
    
    max_size = max_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"파일이 너무 큽니다. 최대: {max_size_mb}MB"
        )
    
    file_ext = os.path.splitext(upload_file.filename)[1]
    if not file_ext:
        file_ext = ".wav"
    
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = UPLOAD_DIR / temp_filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return str(temp_path)


def cleanup_file(file_path: str):
    """임시 파일 삭제"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"파일 삭제 실패: {e}")


# ===== STT API (기존) =====

@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="음성 파일"),
    language: Optional[str] = Form(None, description="언어 코드 (None=자동감지)"),
    word_timestamps: bool = Form(False, description="타임스탬프 포함 여부")
):
    """
    🎤 음성 파일을 텍스트로 변환 (STT만)
    
    **지원 언어:** 99개 언어 (Whisper)
    """
    # ✅ 모듈로 import
    from api import inference
    
    temp_path = None
    start_time = time.time()
    
    try:
        temp_path = save_upload_file(audio_file)
        print(f"📁 파일 저장: {temp_path}")
        
        print(f"🎤 STT 시작...")
        result = inference.whisper_stt.transcribe(
            audio_path=temp_path,
            language=language,
            word_timestamps=word_timestamps
        )
        
        processing_time = time.time() - start_time
        print(f"✅ STT 완료 ({processing_time:.2f}초)")
        
        return STTResponse(
            text=result["text"],
            language=result["language"],
            audio_filename=audio_file.filename,
            processing_time=round(processing_time, 2),
            segments=result["segments"] if word_timestamps else None
        )
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path:
            cleanup_file(temp_path)


# ===== 번역 API (새로 추가!) =====

@router.post("/translate-text", response_model=TranslationResponse)
async def translate_text_only(
    text: str = Form(..., description="번역할 텍스트"),
    source_lang: str = Form("ko", description="원본 언어 (ko, ja, en)"),
    target_lang: str = Form("ja", description="목표 언어 (ko, ja, en)")
):
    """
    📝 → 🌐 텍스트만 번역 (STT 없이)
    
    **지원 언어:** ko ↔ ja (양방향)
    """
    # ✅ 모듈을 import (변수가 아니라!)
    from api import translation
    
    start_time = time.time()
    
    try:
        print(f"🌐 텍스트 번역: {source_lang} → {target_lang}")
        print(f"   원문: {text[:100]}...")
        
        # ✅ 모듈을 통해 접근!
        result = translation.qwen3_translator.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        processing_time = time.time() - start_time
        print(f"✅ 번역 완료 ({processing_time:.2f}초)")
        
        return TranslationResponse(
            original_text=text,
            translated_text=result["translated_text"],
            source_lang=source_lang,
            target_lang=target_lang,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 전체 파이프라인 (새로 추가!) =====

@router.post("/audio-to-translation", response_model=PipelineResponse)
async def audio_to_translation(
    audio_file: UploadFile = File(..., description="음성 파일"),
    target_language: str = Form("ja", description="번역 목표 언어 (ko, ja, en)"),
    stt_language: Optional[str] = Form(None, description="STT 언어 (None=자동감지)")
):
    """
    🎤 → 📝 → 🌐 전체 파이프라인!
    
    **처리 흐름:**
    1. 음성 파일 업로드
    2. Whisper STT (음성 → 텍스트)
    3. Qwen3 번역 (텍스트 → 번역)
    4. 결과 반환
    
    **예시:**
    - 한국어 음성 → 일본어 번역
    - 일본어 음성 → 한국어 번역
    """
    # ✅ 모듈로 import
    from api import inference
    from api import translation
    
    temp_path = None
    total_start = time.time()
    
    try:
        # 1. 파일 저장
        temp_path = save_upload_file(audio_file)
        print(f"📁 파일 저장: {temp_path}")
        
        # 2. STT 수행
        print(f"🎤 STT 시작...")
        stt_start = time.time()
        
        stt_result = inference.whisper_stt.transcribe(
            audio_path=temp_path,
            language=stt_language
        )
        
        transcribed_text = stt_result["text"]
        detected_language = stt_result["language"]
        stt_time = time.time() - stt_start
        
        print(f"✅ STT 완료 ({stt_time:.2f}초)")
        print(f"   감지 언어: {detected_language}")
        print(f"   텍스트: {transcribed_text[:100]}...")
        
        # 3. 번역 수행
        print(f"🌐 번역 시작: {detected_language} → {target_language}")
        translation_start = time.time()
        
        translation_result = translation.qwen3_translator.translate(
            text=transcribed_text,
            source_lang=detected_language,
            target_lang=target_language
        )
        
        translated_text = translation_result["translated_text"]
        translation_time = time.time() - translation_start
        
        print(f"✅ 번역 완료 ({translation_time:.2f}초)")
        print(f"   번역: {translated_text[:100]}...")
        
        # 4. 총 처리 시간
        total_time = time.time() - total_start
        print(f"🎉 전체 파이프라인 완료 ({total_time:.2f}초)")
        
        return PipelineResponse(
            original_audio=audio_file.filename,
            transcribed_text=transcribed_text,
            translated_text=translated_text,
            detected_language=detected_language,
            target_language=target_language,
            processing_time=round(total_time, 2),
            stt_time=round(stt_time, 2),
            translation_time=round(translation_time, 2)
        )
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path:
            cleanup_file(temp_path)


# ===== 헬스 체크 =====

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """시스템 상태 확인"""
    # ✅ 함수 내에서 import (최신 상태 가져오기)
    from api import inference
    from api import translation
    
    # STT 체크
    stt_ok = False
    stt_dev = "unknown"
    if hasattr(inference, 'whisper_stt') and inference.whisper_stt is not None:
        if hasattr(inference.whisper_stt, 'model') and inference.whisper_stt.model is not None:
            stt_ok = True
            stt_dev = getattr(inference.whisper_stt, 'device', 'unknown')
    
    # 번역 체크
    trans_ok = False
    trans_dev = "unknown"
    if hasattr(translation, 'qwen3_translator') and translation.qwen3_translator is not None:
        if hasattr(translation.qwen3_translator, 'model') and translation.qwen3_translator.model is not None:
            trans_ok = True
            trans_dev = getattr(translation.qwen3_translator, 'device', 'unknown')
    
    return HealthResponse(
        status="healthy",
        stt_loaded=stt_ok,
        translator_loaded=trans_ok,
        stt_device=stt_dev,
        translator_device=trans_dev
    )


@router.get("/languages")
async def get_supported_languages():
    """지원 언어 목록"""
    return {
        "stt": {
            "provider": "Whisper",
            "languages": "99개 언어 지원",
            "note": "자동 감지 가능"
        },
        "translation": {
            "provider": "Qwen3-8b LoRA",
            "languages": {
                "ko": "한국어",
                "ja": "日本語",
                "en": "English (실험적)"
            },
            "supported_pairs": [
                "ko → ja",
                "ja → ko"
            ]
        }
    }