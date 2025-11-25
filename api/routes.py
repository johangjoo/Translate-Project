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

# 번역/오디오 파이프라인에서 사용할 클래스들 (요청 시 로딩)
from api.translation import create_translator, TranslationModelType
from api.audio_pipeline import AudioPipeline

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
    save_outputs: bool = Form(True, description="결과 파일 저장"),
    max_speakers: int = Form(2, description="최대 화자 수 (1~10)")
):
   
   
    pipeline: AudioPipeline = AudioPipeline(
        use_gpu=True,
        target_language=language or None,
    )

    # 최대 화자 수 설정 (1~10 범위로 클램프)
    try:
        if max_speakers is not None:
            clamped = max(1, min(10, int(max_speakers)))
            pipeline.max_speakers = clamped
    except Exception:
        # 잘못된 값이 들어와도 기본값(2)을 유지
        pass
    
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
        print(f"파일 업로드: {audio_file.filename}")
        print(f"{'='*60}\n")
        
        work_dir = UPLOAD_DIR / f"work_{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(exist_ok=True)
        
        current_file = temp_path
        
        # 1. 노이즈 제거
        if enable_denoise:
            denoise_start = time.time()
            
            denoised_file = work_dir / f"{Path(audio_file.filename).stem}_denoised.wav"
            
            pipeline.denoise_audio(
                input_file=current_file,
                output_file=str(denoised_file)
            )
            
            timing["denoise"] = time.time() - denoise_start
            result["denoised_filename"] = denoised_file.name
            result["denoise_time"] = round(timing["denoise"], 2)
            
            
            current_file = str(denoised_file)
        else:
            result["denoised_filename"] = None
            result["denoise_time"] = None
        
        # 2. STT + 화자분리
        # 2. STT + 화자분리 섹션에서 수정 (224줄 근처)

        if enable_transcription:
            transcription_start = time.time()
            
            transcript_result = pipeline.transcribe_uploaded_wav(
                wav_path=current_file,
                save_dir=str(work_dir) if save_outputs else None,
                create_srt=create_srt
            )
            
            timing["transcription"] = time.time() - transcription_start
            
            # ✅ 수정: simple 파일 내용 사용
            simple_path = transcript_result.get("simple_path")
            if simple_path and os.path.exists(simple_path):
                with open(simple_path, 'r', encoding='utf-8') as f:
                    result["text"] = f.read()
            else:
                result["text"] = transcript_result["text"]  # fallback
            
            result["detected_language"] = language or "auto"
            result["transcription_time"] = round(timing["transcription"], 2)
            
        else:
            result["text"] = None
            result["detected_language"] = None
            result["transcription_time"] = None
            result["num_speakers"] = None
            result["segments"] = None
        
        total_time = time.time() - total_start
        timing["total"] = total_time
        
        result["processing_time"] = round(total_time, 2)
        result["timing"] = {k: round(v, 2) for k, v in timing.items()}
        
        
        return AudioProcessResponse(**result)

    except Exception as e:
        print(f"\n오류: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path:
            cleanup_file(temp_path)
        # 사용이 끝난 후 모델을 메모리에서 해제하여 VRAM을 확보
        try:
            pipeline.unload_models()
        except Exception:
            pass


# ===== 2. 텍스트 번역 =====

@router.post("/translate-text", response_model=TranslationResponse)
async def translate_text_only(
    text: str = Form(..., description="번역할 텍스트"),
    source_lang: str = Form("ko", description="원본 언어 (ko, ja, en)"),
    target_lang: str = Form("ja", description="목표 언어 (ko, ja, en)"),
    model_type: str = Form("qwen-local", description="번역 모델 타입 (qwen-local, openai, gemini)"),
    api_key: Optional[str] = Form(None, description="API 키 (openai/gemini 사용 시 필수)")
):
   
    start_time = time.time()
    
    try:
        print(f"🌐 텍스트 번역: {source_lang} → {target_lang} (모델: {model_type})")
        print(f"   원문: {text[:100]}...")

        # 모델 타입에 따라 번역기 생성
        translator = None
        
        if model_type == "qwen-local":
            # 로컬 Qwen 모델 경로 찾기 (config.py에서 가져오기)
            from api.config import TRANSLATION_BASE_MODEL
            from pathlib import Path as _Path
            
            # config.py의 경로 사용
            model_path = None
            project_root = _Path(__file__).resolve().parent.parent
            
            # 여러 가능한 경로 시도
            possible_paths = [
                Path(TRANSLATION_BASE_MODEL) / "qwen3-8b-lora-10ratio",
                Path(TRANSLATION_BASE_MODEL),
                project_root / "qwen3-8b-lora-10ratio" / "qwen3-8b-lora-10ratio",
                project_root / "qwen3-8b-lora-10ratio",
            ]
            
            # 경로 찾기
            for path in possible_paths:
                path_obj = Path(path)
                if path_obj.exists() and path_obj.is_dir():
                    # config.json이나 tokenizer.json이 있는지 확인
                    if (path_obj / "config.json").exists() or (path_obj / "tokenizer.json").exists():
                        model_path = path_obj
                        print(f"[OK] 모델 경로 찾음: {model_path}")
                        break
            
            # 모델 경로를 찾지 못한 경우
            if model_path is None:
                error_msg = (
                    f"Qwen 모델을 찾을 수 없습니다.\n"
                    f"시도한 경로:\n"
                )
                for path in possible_paths:
                    error_msg += f"  - {path}\n"
                error_msg += f"\napi/config.py의 TRANSLATION_BASE_MODEL을 확인하세요."
                raise HTTPException(status_code=500, detail=error_msg)
            
            translator = create_translator(
                model_type=TranslationModelType.QWEN_LOCAL,
                model_path=str(model_path),
                use_gpu=True,
                load_in_4bit=True
            )
            
        elif model_type == "openai":
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI 모델 사용 시 api_key가 필요합니다."
                )
            
            # 고정 모델: GPT-5.1
            translator = create_translator(
                model_type=TranslationModelType.OPENAI,
                api_key=api_key,
                model_name="gpt-5.1"
            )
            
        elif model_type == "gemini":
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="Gemini 모델 사용 시 api_key가 필요합니다."
                )
            
            # 고정 모델: Gemini 3 Pro Preview (무료 티어에서는 사용 불가)
            # 무료 티어를 사용하려면 "gemini-1.5-flash"로 변경하세요
            translator = create_translator(
                model_type=TranslationModelType.GEMINI,
                api_key=api_key,
                model_name="gemini-2.5-flash"  # 무료 티어 미지원, 유료 플랜 필요
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 모델 타입: {model_type}. 지원 타입: qwen-local, openai, gemini"
            )
        
        # 모델 로드 및 번역 실행
        translator.load_model()
        try:
            result = translator.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        finally:
            # 번역이 끝나면 모델을 언로드해서 VRAM을 최대한 비워준다
            translator.unload_model()
        
        processing_time = time.time() - start_time
        print(f"✅ 번역 완료 ({processing_time:.2f}초)")
        
        return TranslationResponse(
            original_text=result.original_text,
            translated_text=result.translated_text,
            source_lang=result.source_lang,
            target_lang=result.target_lang,
            audio_filename="N/A",
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 헬스 체크 =====

@router.get("/health", response_model=BasicHealthResponse)
async def health_check():
    """기본 시스템 상태 확인"""
    from api import inference
    
    stt_ok = False
    stt_dev = "unknown"
    if hasattr(inference, 'whisper_stt') and inference.whisper_stt is not None:
        if hasattr(inference.whisper_stt, 'model') and inference.whisper_stt.model is not None:
            stt_ok = True
            stt_dev = getattr(inference.whisper_stt, 'device', 'unknown')
    
    # 번역 모델은 요청 시 로드되므로 항상 사용 가능 상태로 표시
    trans_ok = True  # 모듈화된 번역 시스템은 항상 사용 가능
    trans_dev = "on-demand"  # 요청 시 로드
    
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