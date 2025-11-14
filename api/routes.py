"""
API 라우트 - 간소화 버전
- 오디오 파이프라인 (노이즈 제거 + STT + 화자분리)
- 텍스트 번역
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from pathlib import Path
import os
import uuid
import shutil
import time
import re
from typing import Optional, List, Dict

# models.py에서 import
from .models import (
    TranslationResponse,
    AudioProcessResponse,
    AudioHealthResponse
)

router = APIRouter()

# 업로드 디렉토리
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ===== Pydantic 모델 (routes 전용) =====

class BasicHealthResponse(BaseModel):
    """기본 헬스 체크 응답"""
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


def parse_transcript_segments(transcript_path: str) -> List:
    """전사 파일에서 세그먼트 추출"""
    from .models import AudioSegment
    
    segments = []
    
    try:
        if not os.path.exists(transcript_path):
            return segments
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 형식: [mm:ss - mm:ss] 화자X: 텍스트
            pattern = r'\[(\d+):(\d+\.\d+)\s*-\s*(\d+):(\d+\.\d+)\]\s*(Speaker\d+|화자\d+):\s*(.+)'
            match = re.match(pattern, line)
            
            if match:
                start_min, start_sec, end_min, end_sec, speaker, text = match.groups()
                start = int(start_min) * 60 + float(start_sec)
                end = int(end_min) * 60 + float(end_sec)
                
                segments.append(AudioSegment(
                    start=start,
                    end=end,
                    text=text.strip(),
                    speaker=speaker
                ))
    
    except Exception as e:
        print(f"세그먼트 파싱 실패: {e}")
    
    return segments


# ===== 1. 오디오 파이프라인 =====

@router.post("/audio/process", response_model=AudioProcessResponse)
async def process_audio(
    audio_file: UploadFile = File(..., description="음성 파일"),
    enable_denoise: bool = Form(True, description="노이즈 제거 활성화"),
    enable_transcription: bool = Form(True, description="STT 활성화"),
    enable_diarization: bool = Form(True, description="화자분리 활성화"),
    language: Optional[str] = Form(None, description="언어 코드 (None=자동감지)"),
    create_srt: bool = Form(True, description="SRT 자막 파일 생성"),
    save_outputs: bool = Form(True, description="결과 파일 저장")
):
    """
    🎵 통합 오디오 처리 파이프라인
    
    **기능:**
    1. 🔇 노이즈 제거 (SpeechBrain)
    2. 🎤 STT (Whisper)
    3. 👥 화자 분리
    4. 📝 자막 생성 (SRT)
    
    **옵션:**
    - enable_denoise: 노이즈 제거만 원하면 transcription=false
    - enable_transcription: STT만 원하면 denoise=false
    - enable_diarization: 화자분리 제외하려면 false
    """
    from api import audio_pipeline
    
    pipeline = audio_pipeline.audio_pipeline_instance
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="오디오 파이프라인이 초기화되지 않았습니다."
        )
    
    temp_path = None
    total_start = time.time()
    timing = {}
    
    result = {
        "original_filename": audio_file.filename,
        "denoised": enable_denoise,
        "transcribed": enable_transcription,
        "diarization_enabled": enable_diarization,
    }
    
    try:
        temp_path = save_upload_file(audio_file)
        print(f"\n{'='*60}")
        print(f"📁 파일 업로드: {audio_file.filename}")
        print(f"{'='*60}\n")
        
        work_dir = UPLOAD_DIR / f"work_{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(exist_ok=True)
        
        current_file = temp_path
        
        # 1. 노이즈 제거
        if enable_denoise:
            print("🔇 노이즈 제거 시작...")
            denoise_start = time.time()
            
            denoised_file = work_dir / f"{Path(audio_file.filename).stem}_denoised.wav"
            
            pipeline.denoise_audio(
                input_file=current_file,
                output_file=str(denoised_file)
            )
            
            timing["denoise"] = time.time() - denoise_start
            result["denoised_filename"] = denoised_file.name
            result["denoise_time"] = round(timing["denoise"], 2)
            
            print(f"✅ 노이즈 제거 완료 ({timing['denoise']:.2f}초)\n")
            
            current_file = str(denoised_file)
        else:
            print("⏭️  노이즈 제거 스킵\n")
            result["denoised_filename"] = None
            result["denoise_time"] = None
        
        # 2. STT + 화자분리
        if enable_transcription:
            print("🎤 음성 전사 시작...")
            transcription_start = time.time()
            
            transcript_result = pipeline.transcribe_uploaded_wav(
                wav_path=current_file,
                save_dir=str(work_dir) if save_outputs else None,
                create_srt=create_srt
            )
            
            timing["transcription"] = time.time() - transcription_start
            
            result["text"] = transcript_result["text"]
            result["detected_language"] = language or "auto"
            result["transcription_time"] = round(timing["transcription"], 2)
            
            print(f"✅ 전사 완료 ({timing['transcription']:.2f}초)\n")
            
            if save_outputs:
                result["transcript_path"] = transcript_result.get("transcript_path")
                result["simple_transcript_path"] = transcript_result.get("simple_path")
                result["text_only_path"] = transcript_result.get("text_only_path")
                result["srt_path"] = transcript_result.get("srt_path") if create_srt else None
                
                if result["transcript_path"]:
                    segments = parse_transcript_segments(result["transcript_path"])
                    result["segments"] = segments
                    result["num_speakers"] = len(set(s.speaker for s in segments if s.speaker))
            
        else:
            print("⏭️  음성 전사 스킵\n")
            result["text"] = None
            result["detected_language"] = None
            result["transcription_time"] = None
            result["num_speakers"] = None
            result["segments"] = None
        
        total_time = time.time() - total_start
        timing["total"] = total_time
        
        result["processing_time"] = round(total_time, 2)
        result["timing"] = {k: round(v, 2) for k, v in timing.items()}
        
        print("="*60)
        print(f"🎉 처리 완료! ({total_time:.2f}초)")
        print("="*60 + "\n")
        
        return AudioProcessResponse(**result)
        
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path:
            cleanup_file(temp_path)


# ===== 2. 텍스트 번역 =====

@router.post("/translate-text", response_model=TranslationResponse)
async def translate_text_only(
    text: str = Form(..., description="번역할 텍스트"),
    source_lang: str = Form("ko", description="원본 언어 (ko, ja, en)"),
    target_lang: str = Form("ja", description="목표 언어 (ko, ja, en)")
):
    """
    📝 텍스트 번역
    
    **지원 언어:** ko ↔ ja (양방향)
    """
    from api import translation
    
    start_time = time.time()
    
    try:
        print(f"🌐 텍스트 번역: {source_lang} → {target_lang}")
        print(f"   원문: {text[:100]}...")
        
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
            audio_filename="N/A",
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 헬스 체크 =====

@router.get("/health", response_model=BasicHealthResponse)
async def health_check():
    """기본 시스템 상태 확인"""
    from api import inference
    from api import translation
    
    stt_ok = False
    stt_dev = "unknown"
    if hasattr(inference, 'whisper_stt') and inference.whisper_stt is not None:
        if hasattr(inference.whisper_stt, 'model') and inference.whisper_stt.model is not None:
            stt_ok = True
            stt_dev = getattr(inference.whisper_stt, 'device', 'unknown')
    
    trans_ok = False
    trans_dev = "unknown"
    if hasattr(translation, 'qwen3_translator') and translation.qwen3_translator is not None:
        if hasattr(translation.qwen3_translator, 'model') and translation.qwen3_translator.model is not None:
            trans_ok = True
            trans_dev = getattr(translation.qwen3_translator, 'device', 'unknown')
    
    return BasicHealthResponse(
        status="healthy",
        stt_loaded=stt_ok,
        translator_loaded=trans_ok,
        stt_device=stt_dev,
        translator_device=trans_dev
    )


@router.get("/audio/health", response_model=AudioHealthResponse)
async def audio_health():
    """오디오 파이프라인 상태 확인"""
    from api import audio_pipeline
    
    status = audio_pipeline.get_pipeline_status()
    
    return AudioHealthResponse(
        status="healthy" if status.get("initialized") else "not_initialized",
        initialized=status.get("initialized", False),
        device=status.get("device", "unknown"),
        models=status.get("models", {}),
        gpu_memory=status.get("gpu_memory")
    )


@router.get("/audio/memory")
async def memory_stats():
    """GPU 메모리 상태"""
    from api import audio_pipeline
    return audio_pipeline.get_memory_stats()


@router.get("/languages")
async def get_supported_languages():
    """지원 언어 목록"""
    return {
        "stt": {
            "provider": "Whisper",
            "languages": "99개 언어 지원"
        },
        "translation": {
            "provider": "Qwen3-8b LoRA",
            "languages": {
                "ko": "한국어",
                "ja": "日本語",
                "en": "English"
            }
        }
    }