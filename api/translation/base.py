"""
번역기 베이스 클래스 및 공통 타입 정의
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """번역 결과 데이터 클래스"""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = "unknown"


class BaseTranslator(ABC):
    """번역기 베이스 클래스 (추상 인터페이스)"""
    
    def __init__(self, model_name: str):
        """
        Args:
            model_name: 모델 이름 (예: "qwen-local", "openai", "gemini")
        """
        self.model_name = model_name
        self._loaded = False
    
    @abstractmethod
    def load_model(self, **kwargs):
        """모델 로딩 (구현 필수)"""
        pass
    
    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "ja",
        **kwargs
    ) -> TranslationResult:
        """
        텍스트 번역 (구현 필수)
        
        Args:
            text: 번역할 텍스트
            source_lang: 원본 언어 (ko, ja, en)
            target_lang: 목표 언어 (ko, ja, en)
            **kwargs: 모델별 추가 옵션
        
        Returns:
            TranslationResult: 번역 결과
        """
        pass
    
    def unload_model(self):
        """모델 언로드 (선택적 구현)"""
        self._loaded = False
    
    def get_memory_stats(self) -> Dict[str, float]:
        """메모리 사용량 조회 (로컬 모델만 구현)"""
        return {"message": "Not available for this model type"}
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부"""
        return self._loaded
    
    def _detect_format(self, text: str) -> str:
        """
        텍스트 형식 자동 감지
        
        Returns:
            "transcript": 트랜스크립트 형식 ([mm:ss] 화자: 내용)
            "multiline": 여러 줄 텍스트 (3줄 이상)
            "single": 일반 텍스트
        """
        import re
        
        lines = text.strip().split('\n')
        if not lines:
            return "single"
        
        # 트랜스크립트 형식 감지: [타임스탬프] 화자: 내용
        transcript_pattern = r'^(\[[\d:\.]+\])?\s*(화자\d+|Speaker\d+|[^:]+):\s*.+$'
        first_line = lines[0].strip()
        is_transcript = bool(re.match(transcript_pattern, first_line))
        
        if is_transcript:
            return "transcript"
        
        # 여러 줄 텍스트 (3줄 이상)
        if len(lines) >= 3:
            return "multiline"
        
        return "single"

