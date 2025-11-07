from pydantic import BaseModel, Field
from typing import Optional

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