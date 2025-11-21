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
        """애니/영상 자막 번역용 시스템 프롬프트"""
        return f"""
You are a professional subtitle translator specializing in {source_full} → {target_full}.
You mainly translate anime, movies, dramas, games, and YouTube videos.

General rules:
- Translate ONLY the utterance marked [CURRENT_LINE].
- Use [PREVIOUS_LINES] and [NEXT_LINES] only to understand context
  (who is speaking, who pronouns refer to, running jokes, relationship, tone).
- 1 input line MUST correspond to 1 output line. Do not merge or split lines.
- Keep the translation concise and readable as on-screen subtitles.
- Preserve speaker names, timecodes, brackets, emoji, and sound effects when meaningful.
- Preserve honorifics or speech level implied in the source (polite, casual, rude, etc.)
- Do NOT add any explanations, notes, or commentary.
- Output ONLY the translated text for [CURRENT_LINE] in {target_full}.
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

    def _build_user_prompt_subtitle(
        self,
        source_full: str,
        target_full: str,
        previous_lines: str,
        current_line: str,
        next_lines: str,
    ) -> str:
        """자막 컨텍스트용 유저 프롬프트"""
        prev_block = previous_lines.strip() if previous_lines.strip() else "(none)"
        next_block = next_lines.strip() if next_lines.strip() else "(none)"

        return f"""
[SOURCE_LANGUAGE]: {source_full}
[TARGET_LANGUAGE]: {target_full}

[PREVIOUS_LINES]
{prev_block}

[CURRENT_LINE]
{current_line}

[NEXT_LINES]
{next_block}

Task:
Translate [CURRENT_LINE] into {target_full} as a natural subtitle.
Return ONLY the translation of [CURRENT_LINE].
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
        """여러 줄 텍스트 번역 (애니/유튜브 스크립트 가정, 라인 컨텍스트 사용)"""
        lines = text.strip().split("\n")
        translated_lines = []
        total_input_tokens = 0
        total_output_tokens = 0

        source_full = self._get_lang_name(source_lang)
        target_full = self._get_lang_name(target_lang)

        logger.info(f"줄 단위 번역 (컨텍스트): {len(lines)}줄")

        for idx, line in enumerate(lines):
            current = line.strip()

            if not current:
                translated_lines.append("")
                continue

            # 윗/아랫줄 컨텍스트 구성
            start_idx = max(0, idx - self.CONTEXT_WINDOW_LINES)
            end_idx = min(len(lines), idx + self.CONTEXT_WINDOW_LINES + 1)

            previous_lines = "\n".join(lines[start_idx:idx])
            next_lines = "\n".join(lines[idx + 1:end_idx])

            system_prompt = self._build_system_prompt_subtitle(source_full, target_full)
            user_prompt = self._build_user_prompt_subtitle(
                source_full,
                target_full,
                previous_lines=previous_lines,
                current_line=current,
                next_lines=next_lines,
            )

            translated_text, in_tok, out_tok = self._call_openai(
                system_prompt, user_prompt, temperature, max_output_tokens=512
            )

            translated_lines.append(translated_text)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            logger.debug(f"  {idx + 1}/{len(lines)} 줄 변역 완료")

        logger.info(f"[OK] 줄 단위 번역 완료: {len(lines)}줄")

        return TranslationResult(
            original_text=text,
            translated_text="\n".join(translated_lines),
            source_lang=source_lang,
            target_lang=target_lang,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
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
        """트랜스크립트 번역: [타임스탬프] 화자: 내용  형식을 유지하면서 번역"""
        lines = text.strip().split("\n")
        translated_lines = []
        total_input_tokens = 0
        total_output_tokens = 0

        # [타임스탬프] Speaker: 내용
        pattern = r'^(\[[\d:\.]+\])?\s*(화자\d+|Speaker\d+|[^:]+):\s*(.+)$'

        source_full = self._get_lang_name(source_lang)
        target_full = self._get_lang_name(target_lang)

        logger.info(f"트랜스크립트 번역 (컨텍스트): {len(lines)}줄")

        for idx, raw_line in enumerate(lines):
            line = raw_line.strip()

            if not line:
                translated_lines.append("")
                continue

            # 컨텍스트 윈도우 계산
            start_idx = max(0, idx - self.CONTEXT_WINDOW_LINES)
            end_idx = min(len(lines), idx + self.CONTEXT_WINDOW_LINES + 1)

            previous_lines = "\n".join(lines[start_idx:idx])
            next_lines = "\n".join(lines[idx + 1:end_idx])

            match = re.match(pattern, line)

            if match:
                timestamp = match.group(1) or ""
                speaker = match.group(2)
                content = match.group(3).strip()

                system_prompt = self._build_system_prompt_subtitle(source_full, target_full)
                user_prompt = self._build_user_prompt_subtitle(
                    source_full,
                    target_full,
                    previous_lines=previous_lines,
                    current_line=content,
                    next_lines=next_lines,
                )

                try:
                    translated_text, in_tok, out_tok = self._call_openai(
                        system_prompt, user_prompt, temperature, max_output_tokens=512
                    )

                    if timestamp:
                        reconstructed = f"{timestamp} {speaker}: {translated_text}"
                    else:
                        reconstructed = f"{speaker}: {translated_text}"

                    translated_lines.append(reconstructed)
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok

                    logger.debug(f"  {idx + 1}/{len(lines)} [{speaker}] 번역 완료")

                except Exception as e:
                    logger.error(f"  {idx + 1}번째 줄 실패: {e}")
                    translated_lines.append(f"[번역 실패] {line}")
            else:
                # 패턴 불일치: 일반 줄로 취급
                system_prompt = self._build_system_prompt_subtitle(source_full, target_full)
                user_prompt = self._build_user_prompt_subtitle(
                    source_full,
                    target_full,
                    previous_lines=previous_lines,
                    current_line=line,
                    next_lines=next_lines,
                )

                try:
                    translated_text, in_tok, out_tok = self._call_openai(
                        system_prompt, user_prompt, temperature, max_output_tokens=512
                    )
                    translated_lines.append(translated_text)
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                except Exception as e:
                    logger.error(f"  {idx + 1}번째 줄 실패(패턴 불일치): {e}")
                    translated_lines.append(f"[번역 실패] {line}")

        logger.info(f"[OK] 트랜스크립트 번역 완료: {len(lines)}줄")

        return TranslationResult(
            original_text=text,
            translated_text="\n".join(translated_lines),
            source_lang=source_lang,
            target_lang=target_lang,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model_name=f"{self.model_name}:{self.model}",
        )
