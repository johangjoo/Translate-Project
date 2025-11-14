from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# ===== 기존 모델 =====

class TranslationResponse(BaseModel):
    """번역 응답 모델"""
    original_text: str = Field(..., description="원본 텍스트 (STT 결과)")
    translated_text: str = Field(..., description="번역된 텍스트")
    source_lang: str = Field(default="en", description="원본 언어")
    target_lang: str = Field(default="ko", description="목표 언어")
    audio_filename: str = Field(..., description="업로드된 파일명")
    processing_time: float = Field(..., description="처리 시간(초)")


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str


# ===== Audio Pipeline 모델 =====

class AudioSegment(BaseModel):
    """오디오 세그먼트 (화자 정보 포함)"""
    start: float = Field(..., description="시작 시간(초)")
    end: float = Field(..., description="종료 시간(초)")
    text: str = Field(..., description="전사 텍스트")
    speaker: Optional[str] = Field(None, description="화자 ID (Speaker1, Speaker2, ...)")


class AudioProcessResponse(BaseModel):
    """통합 오디오 처리 응답"""
    # 기본 정보
    original_filename: str = Field(..., description="원본 파일명")
    processing_time: float = Field(..., description="총 처리 시간(초)")
    
    # 노이즈 제거 결과
    denoised: bool = Field(..., description="노이즈 제거 수행 여부")
    denoised_filename: Optional[str] = Field(None, description="노이즈 제거된 파일명")
    denoise_time: Optional[float] = Field(None, description="노이즈 제거 시간(초)")
    
    # STT 결과
    transcribed: bool = Field(..., description="전사 수행 여부")
    text: Optional[str] = Field(None, description="전체 전사 텍스트")
    detected_language: Optional[str] = Field(None, description="감지된 언어")
    transcription_time: Optional[float] = Field(None, description="전사 시간(초)")
    
    # 화자분리 결과
    diarization_enabled: bool = Field(..., description="화자분리 수행 여부")
    num_speakers: Optional[int] = Field(None, description="감지된 화자 수")
    segments: Optional[List[AudioSegment]] = Field(None, description="화자별 세그먼트")
    
    # 출력 파일
    transcript_path: Optional[str] = Field(None, description="전사 파일 경로")
    simple_transcript_path: Optional[str] = Field(None, description="간단 전사 파일 경로")
    srt_path: Optional[str] = Field(None, description="SRT 자막 파일 경로")
    text_only_path: Optional[str] = Field(None, description="텍스트 전용 파일 경로")
    
    # 처리 단계별 시간
    timing: Dict[str, float] = Field(..., description="단계별 처리 시간")


class AudioHealthResponse(BaseModel):
    """오디오 파이프라인 상태"""
    status: str
    initialized: bool
    device: str
    models: Dict[str, bool]
    gpu_memory: Optional[Dict[str, float]] = None