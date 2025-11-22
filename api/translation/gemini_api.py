"""
Google Gemini API 번역 모델 (System Instruction & Safety Settings 적용)
"""

import logging
from typing import Optional, Dict, Tuple
import re

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .base import BaseTranslator, TranslationResult

logger = logging.getLogger(__name__)

class GeminiTranslator(BaseTranslator):
    """Google Gemini API 기반 번역 모델"""

    # 문맥 윈도우 (앞뒤 몇 줄을 볼 것인가)
    CONTEXT_WINDOW_LINES = 2
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Args:
            api_key: Gemini API 키
            model: gemini-2.5-flash 
        """
        super().__init__("gemini")
        
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai 패키지가 없습니다. "
                "pip install google-generativeai"
            )
        
        if not api_key:
            raise ValueError("Gemini API 키가 필요합니다.")
        
        self.api_key = api_key
        self.model_name_str = model  # 실제 모델명 문자열
        self.client: Optional[genai.GenerativeModel] = None
        
        logger.info(f"GeminiTranslator 초기화 - 모델: {self.model_name_str}")
    
    # ------------------------------------------------------------------
    # 시스템 프롬프트 (Gemini는 초기화 시점에 넣는 것을 권장)
    # ------------------------------------------------------------------
    def _get_system_instruction(self) -> str:
        """번역가 페르소나 정의"""
        return """
You are a professional translator specializing in Korean-Japanese translation.
Your task is to translate the text in [CURRENT_LINE] from [Source Language] to [Target Language].

CRITICAL RULES:
1. You MUST translate the text. Never return the original text unchanged.
2. Translate ONLY the text inside [CURRENT_LINE]. Do not translate [PREVIOUS_LINES] or [NEXT_LINES].
3. Do not add explanations, notes, greetings, or any additional text.
4. Maintain the speaker's tone, honorifics, and style inferred from the context.
5. If the input is a sound effect or proper noun that shouldn't be translated, transliterate it appropriately.
6. Output ONLY the translated text, nothing else.

Example:
Input: [CURRENT_LINE] 안녕하세요
Output: こんにちは

If you cannot translate, return "[TRANSLATION_ERROR]" instead of the original text.
""".strip()

    def load_model(self, **kwargs):
        """Gemini 클라이언트 및 모델 설정"""
        if self.client is not None:
            return
        
        try:
            genai.configure(api_key=self.api_key)
            
            # 안전 설정 (번역 거부 방지: BLOCK_NONE 권장)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # 모델 생성 시 system_instruction 주입
            self.client = genai.GenerativeModel(
                model_name=self.model_name_str,
                system_instruction=self._get_system_instruction(),
                safety_settings=safety_settings
            )
            self._loaded = True
            logger.info("[OK] Gemini 클라이언트 로드 완료 (System Instruction 적용됨)")
            
        except Exception as e:
            logger.error(f"[ERROR] Gemini 초기화 실패: {e}")
            raise

    def unload_model(self):
        self.client = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 실제 번역 실행
    # ------------------------------------------------------------------
    def translate(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "ja",
        temperature: float = 0.3,
        **kwargs
    ) -> TranslationResult:
        
        if self.client is None:
            self.load_model()

        # 형식 감지 및 분기
        format_type = self._detect_format(text)
        
        if format_type == "transcript":
            return self._translate_lines(text, source_lang, target_lang, temperature, is_transcript=True)
        elif format_type == "multiline":
            return self._translate_lines(text, source_lang, target_lang, temperature, is_transcript=False)
        else:
            # 단일 문장도 문맥 없는 리스트로 처리
            return self._translate_lines(text, source_lang, target_lang, temperature, is_transcript=False)

    def _translate_lines(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        temperature: float,
        is_transcript: bool
    ) -> TranslationResult:
        """줄 단위 문맥 번역 (핵심 로직)"""
        lines = text.strip().split('\n')
        translated_lines = []
        total_in_tokens = 0
        total_out_tokens = 0

        # 언어 이름 매핑
        lang_map = {"ko": "Korean", "ja": "Japanese", "en": "English"}
        s_full = lang_map.get(source_lang, source_lang)
        t_full = lang_map.get(target_lang, target_lang)

        # Regex (트랜스크립트용)
        pattern = r'^(\[[\d:\.]+\])?\s*([^:]+?)\s*:\s*(.+)$'

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
        )

        logger.info(f"Gemini 번역 시작 ({len(lines)}줄) - {s_full} -> {t_full}")

        for i, line in enumerate(lines):
            current_line = line.strip()
            if not current_line:
                translated_lines.append("")
                continue

            # 1. 문맥 추출 (앞뒤 N줄)
            start = max(0, i - self.CONTEXT_WINDOW_LINES)
            end = min(len(lines), i + self.CONTEXT_WINDOW_LINES + 1)
            prev_lines = "\n".join(lines[start:i])
            next_lines = "\n".join(lines[i+1:end])

            # 2. 트랜스크립트 파싱
            speaker_prefix = ""
            content_to_translate = current_line
            
            if is_transcript:
                match = re.match(pattern, current_line)
                if match:
                    timestamp = match.group(1) or ""
                    speaker = match.group(2)
                    content = match.group(3)
                    # 번역할 내용은 content, 앞부분은 나중에 붙임
                    speaker_prefix = f"{timestamp} {speaker}: " if timestamp else f"{speaker}: "
                    content_to_translate = content.strip()

            # 3. Gemini 전용 유저 프롬프트 구성
            user_prompt = f"""
[Source Language]: {s_full}
[Target Language]: {t_full}

[PREVIOUS_LINES]
{prev_lines if prev_lines else "(None)"}

[CURRENT_LINE]
{content_to_translate}

[NEXT_LINES]
{next_lines if next_lines else "(None)"}
"""
            try:
                # API 호출
                response = self.client.generate_content(
                    user_prompt,
                    generation_config=generation_config
                )
                
                # 결과 처리 - Gemini 3 Pro Preview 호환성 개선
                trans_text = None
                
                # 응답이 있는지 확인
                if hasattr(response, 'text') and response.text:
                    trans_text = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    # candidates에서 텍스트 추출
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            # parts에서 텍스트 추출
                            text_parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                            if text_parts:
                                trans_text = " ".join(text_parts).strip()
                
                # 번역 결과가 없거나 원문과 동일한 경우
                if not trans_text or trans_text == content_to_translate:
                    logger.warning(f"  {i+1}번째 줄: 번역 결과가 없거나 원문과 동일함. 원문: {content_to_translate[:50]}...")
                    # 원문을 그대로 사용하지 않고 재시도 또는 에러 표시
                    trans_text = f"[번역 실패: {content_to_translate}]"
                
                # 메타데이터에서 토큰 수 가져오기
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    total_in_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                    total_out_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)

                # 트랜스크립트 재조립
                if is_transcript and speaker_prefix:
                    translated_lines.append(f"{speaker_prefix}{trans_text}")
                else:
                    translated_lines.append(trans_text)
                
                logger.debug(f"  {i+1}/{len(lines)} 완료: {content_to_translate[:30]}... -> {trans_text[:30]}...")

            except Exception as e:
                error_str = str(e)
                logger.error(f"  {i+1}번째 줄 실패: {error_str}")
                logger.error(f"    원문: {content_to_translate[:100]}")
                
                # 쿼터 초과 에러인 경우 명확한 메시지
                if "429" in error_str or "quota" in error_str.lower() or "Quota exceeded" in error_str:
                    error_msg = "[ERROR] Gemini API 쿼터 초과: gemini-3-pro-preview는 무료 티어에서 사용할 수 없습니다. gemini-1.5-flash를 사용하세요."
                    logger.error(error_msg)
                    translated_lines.append(f"[번역 실패: API 쿼터 초과 - gemini-3-pro-preview는 무료 티어 미지원]")
                else:
                    # 실패 시 에러 표시 (원문 그대로 반환하지 않음)
                    translated_lines.append(f"[번역 오류: {error_str[:50]}]")

        return TranslationResult(
            original_text=text,
            translated_text="\n".join(translated_lines),
            source_lang=source_lang,
            target_lang=target_lang,
            input_tokens=total_in_tokens,
            output_tokens=total_out_tokens,
            model_name=f"gemini:{self.model_name_str}"
        )