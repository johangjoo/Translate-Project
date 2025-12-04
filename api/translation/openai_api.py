"""
OpenAI API 번역 모델 (애니/유튜브 스크립트 컨텍스트 지원 버전)
"""

import logging
from typing import Optional, Dict, Tuple
import re

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseTranslator, TranslationResult

logger = logging.getLogger(__name__)


class OpenAITranslator(BaseTranslator):
    """OpenAI API 기반 번역 모델"""

    # 자막 컨텍스트 윈도우 (위/아래 몇 줄까지 볼지)
    CONTEXT_WINDOW_LINES = 2

    def __init__(self, api_key: str, model: str = "gpt-5.1"):
        """
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델 (예: gpt-5.1, gpt-4.1, gpt-4.1-mini 등)
        """
        super().__init__("openai")

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai 패키지가 설치되지 않았습니다. "
                "pip install openai 로 설치해주세요."
            )

        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다.")

        self.api_key = api_key
        self.model = model
        self.client: Optional[OpenAI] = None

        logger.info(f"OpenAITranslator 초기화 - 모델: {self.model}")

    # ------------------------------------------------------------------
    # 초기화 / 정리
    # ------------------------------------------------------------------
    def load_model(self, **kwargs):
        """OpenAI 클라이언트 초기화"""
        if self.client is not None:
            logger.warning("클라이언트가 이미 초기화되어 있습니다.")
            self._loaded = True
            return

        try:
            self.client = OpenAI(api_key=self.api_key)
            self._loaded = True
            logger.info("[OK] OpenAI 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"[ERROR] OpenAI 클라이언트 초기화 실패: {e}")
            raise

    def unload_model(self):
        """클라이언트 정리"""
        self.client = None
        self._loaded = False
        logger.info("OpenAI 클라이언트 정리 완료")

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------
    def translate(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "ja",
        temperature: float = 0.3,
        **kwargs
    ) -> TranslationResult:
        """텍스트 번역 (형식 자동 감지)"""
        if self.client is None:
            raise RuntimeError("클라이언트가 초기화되지 않았습니다. load_model()을 먼저 호출하세요.")

        # 형식 자동 감지 (BaseTranslator 쪽에 있다고 가정)
        format_type = self._detect_format(text)

        if format_type == "transcript":
            logger.info("📋 자동 감지: 트랜스크립트 형식")
            result = self._translate_transcript(text, source_lang, target_lang, temperature)
        elif format_type == "multiline":
            logger.info("📝 자동 감지: 여러 줄 텍스트")
            result = self._translate_multiline(text, source_lang, target_lang, temperature)
        else:
            logger.info("💬 자동 감지: 일반 텍스트")
            result = self._translate_single(text, source_lang, target_lang, temperature)

        return result

    # ------------------------------------------------------------------
    # 내부 헬퍼: 공통 프롬프트/호출
    # ------------------------------------------------------------------
    @staticmethod
    def _get_lang_name(lang_code: str) -> str:
        lang_map = {
            "ko": "Korean",
            "ja": "Japanese",
            "en": "English",
        }
        return lang_map.get(lang_code.lower(), lang_code)

    def _build_system_prompt_basic(self, source_full: str, target_full: str) -> str:
        """일반 단일 문장/문단 번역용 시스템 프롬프트"""
        return (
            f"You are a professional translator specializing in {source_full} to {target_full} translation. "
            "Translate ONLY the given text accurately without any explanations, notes, or additional content. "
            "Do not mix other languages. Output only the translated text."
        )

    def _build_system_prompt_subtitle(self, source_full: str, target_full: str) -> str:
        """애니/영상 자막 번역용 시스템 프롬프트 (전체 텍스트 번역)"""
        return f"""
You are a professional subtitle translator specializing in {source_full} → {target_full}.
You mainly translate anime, movies, dramas, games, and YouTube videos.

General rules:
- Translate the ENTIRE text while maintaining the original line structure.
- Understand the full context to ensure consistent translation of characters, relationships, and running themes.
- Maintain the exact number of lines: 1 input line MUST correspond to 1 output line. Do not merge or split lines.
- Keep the translation concise and readable as on-screen subtitles.
- Preserve speaker names, timecodes, brackets, emoji, and sound effects when meaningful.
- Preserve honorifics or speech level implied in the source (polite, casual, rude, etc.)
- Ensure consistency in character names, pronouns, and terminology throughout the entire text.
- Do NOT add any explanations, notes, or commentary.
- Output ONLY the translated text in {target_full}, maintaining the same line structure as the input.
""".strip()

    def _build_user_prompt_basic(
        self,
        text: str,
        source_full: str,
        target_full: str,
    ) -> str:
        """일반 번역용 유저 프롬프트"""
        return (
            f"Translate the following text from {source_full} to {target_full}:\n\n{text}"
        )

    def _build_user_prompt_full_text(
        self,
        source_full: str,
        target_full: str,
        full_text: str,
    ) -> str:
        """전체 텍스트 번역용 유저 프롬프트"""
        return f"""
[SOURCE_LANGUAGE]: {source_full}
[TARGET_LANGUAGE]: {target_full}

[FULL_TEXT_TO_TRANSLATE]
{full_text}

Task:
Translate the entire text above from {source_full} to {target_full}.
Maintain the exact line structure - each line should be translated separately but with full context awareness.
Return ONLY the translated text, preserving the same number of lines and structure.
""".strip()

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int = 512,
    ) -> Tuple[str, int, int]:
        """Responses API 공통 호출 부분"""
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            # 텍스트 추출
            translated_text = getattr(response, "output_text", None)
            if not translated_text:
                # 안전하게 fallback
                translated_text = (
                    response.output[0].content[0].text.strip()
                    if response.output and response.output[0].content
                    else ""
                )
            translated_text = translated_text.strip()

            # 토큰 사용량
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

            return translated_text, input_tokens, output_tokens

        except Exception as e:
            logger.error(f"[ERROR] OpenAI 번역 호출 실패: {e}")
            raise

    # ------------------------------------------------------------------
    # 단일 텍스트 번역 (컨텍스트 없음)
    # ------------------------------------------------------------------
    def _translate_single(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        temperature: float,
    ) -> TranslationResult:
        """단일 텍스트 번역 (자막 컨텍스트 X)"""
        logger.info(f"번역 시작 (단일): {source_lang} → {target_lang}")

        source_full = self._get_lang_name(source_lang)
        target_full = self._get_lang_name(target_lang)

        system_prompt = self._build_system_prompt_basic(source_full, target_full)
        user_prompt = self._build_user_prompt_basic(text, source_full, target_full)

        translated_text, input_tokens, output_tokens = self._call_openai(
            system_prompt, user_prompt, temperature, max_output_tokens=4096
        )

        logger.info(f"[OK] 번역 완료: {len(translated_text)} 글자")

        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=f"{self.model_name}:{self.model}",
        )

    # ------------------------------------------------------------------
    # 여러 줄 텍스트 번역 (자막 컨텍스트 ON)
    # ------------------------------------------------------------------
    def _translate_multiline(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        temperature: float,
    ) -> TranslationResult:
        """여러 줄 텍스트 번역 (전체 텍스트를 한 번에 번역하여 문맥 파악)"""
        source_full = self._get_lang_name(source_lang)
        target_full = self._get_lang_name(target_lang)

        lines = text.strip().split("\n")
        logger.info(f"전체 텍스트 번역: {len(lines)}줄")

        system_prompt = self._build_system_prompt_subtitle(source_full, target_full)
        user_prompt = self._build_user_prompt_full_text(
            source_full,
            target_full,
            text,
        )

        # 전체 텍스트를 한 번에 번역
        translated_text, input_tokens, output_tokens = self._call_openai(
            system_prompt, user_prompt, temperature, max_output_tokens=4096
        )

        # 줄 수가 일치하는지 확인하고 조정
        translated_lines = translated_text.strip().split("\n")
        original_lines = text.strip().split("\n")
        
        # 줄 수가 다르면 경고
        if len(translated_lines) != len(original_lines):
            logger.warning(
                f"번역된 줄 수({len(translated_lines)})가 원본 줄 수({len(original_lines)})와 다릅니다. "
                "원본 줄 구조를 유지하도록 조정합니다."
            )
            # 원본 줄 수에 맞춰 조정
            if len(translated_lines) < len(original_lines):
                # 부족한 줄은 빈 줄로 채움
                translated_lines.extend([""] * (len(original_lines) - len(translated_lines)))
            else:
                # 초과한 줄은 병합
                translated_lines = translated_lines[:len(original_lines)]

        logger.info(f"[OK] 전체 텍스트 번역 완료: {len(original_lines)}줄")

        return TranslationResult(
            original_text=text,
            translated_text="\n".join(translated_lines),
            source_lang=source_lang,
            target_lang=target_lang,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=f"{self.model_name}:{self.model}",
        )

    # ------------------------------------------------------------------
    # 트랜스크립트 번역 (타임스탬프/화자 유지 + 컨텍스트)
    # ------------------------------------------------------------------
    def _translate_transcript(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        temperature: float,
    ) -> TranslationResult:
        """트랜스크립트 번역: [타임스탬프] 화자: 내용  형식을 유지하면서 전체 텍스트를 한 번에 번역"""
        lines = text.strip().split("\n")
        # [타임스탬프] Speaker: 내용
        pattern = r'^(\[[\d:\.]+\])?\s*(화자\d+|Speaker\d+|[^:]+):\s*(.+)$'

        source_full = self._get_lang_name(source_lang)
        target_full = self._get_lang_name(target_lang)

        logger.info(f"트랜스크립트 전체 번역: {len(lines)}줄")

        # 타임스탬프와 화자 정보 추출
        transcript_parts = []
        for line in lines:
            if not line.strip():
                transcript_parts.append(("", "", ""))
                continue
            
            match = re.match(pattern, line)
            if match:
                timestamp = match.group(1) or ""
                speaker = match.group(2)
                content = match.group(3).strip()
                transcript_parts.append((timestamp, speaker, content))
            else:
                # 패턴 불일치: 전체를 내용으로 취급
                transcript_parts.append(("", "", line.strip()))

        # 내용만 추출하여 번역
        content_lines = []
        for timestamp, speaker, content in transcript_parts:
            if timestamp and speaker:
                content_lines.append(content)
            elif content:
                content_lines.append(content)
            else:
                content_lines.append("")

        content_text = "\n".join(content_lines)

        # 전체 내용을 한 번에 번역
        system_prompt = self._build_system_prompt_subtitle(source_full, target_full)
        user_prompt = self._build_user_prompt_full_text(
            source_full,
            target_full,
            content_text,
        )

        try:
            translated_content, input_tokens, output_tokens = self._call_openai(
                system_prompt, user_prompt, temperature, max_output_tokens=4096
            )

            # 번역된 내용을 줄 단위로 분리
            translated_content_lines = translated_content.strip().split("\n")
            
            # 원본 구조에 맞춰 재조립
            translated_lines = []
            for idx, (timestamp, speaker, _) in enumerate(transcript_parts):
                if not timestamp and not speaker and not transcript_parts[idx][2]:
                    # 빈 줄
                    translated_lines.append("")
                elif idx < len(translated_content_lines):
                    translated_text = translated_content_lines[idx].strip()
                    if timestamp and speaker:
                        reconstructed = f"{timestamp} {speaker}: {translated_text}"
                    elif speaker:
                        reconstructed = f"{speaker}: {translated_text}"
                    else:
                        reconstructed = translated_text
                    translated_lines.append(reconstructed)
                else:
                    # 번역 결과가 부족한 경우
                    logger.warning(f"  {idx + 1}번째 줄: 번역 결과 부족")
                    translated_lines.append(f"[번역 실패] {lines[idx]}")

            logger.info(f"[OK] 트랜스크립트 번역 완료: {len(lines)}줄")

            return TranslationResult(
                original_text=text,
                translated_text="\n".join(translated_lines),
                source_lang=source_lang,
                target_lang=target_lang,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_name=f"{self.model_name}:{self.model}",
            )

        except Exception as e:
            logger.error(f"[ERROR] 트랜스크립트 번역 실패: {e}")
            raise
