"""
번역기 팩토리 - 모델 타입에 따라 적절한 번역기 생성
"""

from enum import Enum
from typing import Optional
import logging

from .base import BaseTranslator
from .qwen_local import QwenLocalTranslator
from .openai_api import OpenAITranslator
from .gemini_api import GeminiTranslator

logger = logging.getLogger(__name__)


class TranslationModelType(str, Enum):
    """번역 모델 타입"""
    QWEN_LOCAL = "qwen-local"
    OPENAI = "openai"
    GEMINI = "gemini"


def create_translator(
    model_type: str | TranslationModelType,
    model_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    use_gpu: bool = True,
    load_in_4bit: bool = True,
    **kwargs
) -> BaseTranslator:
    """
    모델 타입에 따라 적절한 번역기 생성
    
    Args:
        model_type: 모델 타입 ("qwen-local", "openai", "gemini")
        model_path: 로컬 모델 경로 (qwen-local만 필요)
        api_key: API 키 (openai, gemini만 필요)
        model_name: API 모델 이름 (openai: "gpt-5.1", gemini: gemini-3-pro-preview")
        use_gpu: GPU 사용 여부 (qwen-local만 적용)
        load_in_4bit: 4bit 양자화 (qwen-local만 적용)
        **kwargs: 모델별 추가 옵션
    
    Returns:
        BaseTranslator: 생성된 번역기 인스턴스
    
    Examples:
        >>> # 로컬 Qwen 모델
        >>> translator = create_translator(
        ...     "qwen-local",
        ...     model_path="/path/to/model"
        ... )
        
        >>> # OpenAI API
        >>> translator = create_translator(
        ...     "openai",
        ...     api_key="sk-...",
        ...     model_name="gpt-5.1"
        ... )
        
        >>> # Gemini API
        >>> translator = create_translator(
        ...     "gemini",
        ...     api_key="AIza...",
        ...     model_name="gemini-1.5-flash"
        ... )
    """
    # 문자열을 Enum으로 변환
    if isinstance(model_type, str):
        try:
            model_type = TranslationModelType(model_type.lower())
        except ValueError:
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}. "
                f"지원 타입: {[e.value for e in TranslationModelType]}"
            )
    
    logger.info(f"번역기 생성: {model_type.value}")
    
    if model_type == TranslationModelType.QWEN_LOCAL:
        if not model_path:
            raise ValueError("qwen-local 모델은 model_path가 필요합니다.")
        
        translator = QwenLocalTranslator(
            model_path=model_path,
            use_gpu=use_gpu,
            load_in_4bit=load_in_4bit
        )
        
    elif model_type == TranslationModelType.OPENAI:
        if not api_key:
            raise ValueError("openai 모델은 api_key가 필요합니다.")
        
        model = model_name or "gpt-5.1"
        translator = OpenAITranslator(
            api_key=api_key,
            model=model
        )
        
    elif model_type == TranslationModelType.GEMINI:
        if not api_key:
            raise ValueError("gemini 모델은 api_key가 필요합니다.")
        
        model = model_name or "gemini-2.5-flash"
        translator = GeminiTranslator(
            api_key=api_key,
            model=model
        )
        
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    logger.info(f"[OK] 번역기 생성 완료: {translator.model_name}")
    return translator

