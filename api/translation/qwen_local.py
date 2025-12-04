
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from pathlib import Path
from typing import Optional, Dict
import re

from .base import BaseTranslator, TranslationResult

logger = logging.getLogger(__name__)


class QwenLocalTranslator(BaseTranslator):
    """qwen 14b 사용 (8b도 지원)"""
    
    # 14B 모델 최적화 설정 (24GB GPU 기준, 4bit 양자화 사용 시 약 8-10GB)
    MAX_INPUT_LENGTH = 4096
    MAX_OUTPUT_CAP = 4096
    MIN_OUTPUT_TOKENS = 512
    
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        load_in_4bit: bool = True
    ):
        """
        Args:
            model_path: LoRA 모델 경로
            use_gpu: GPU 사용 여부
            load_in_4bit: 4bit 양자화 사용 여부
        """
        super().__init__("qwen-local")
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"QwenLocalTranslator 초기화 - 디바이스: {self.device}")
        if self.use_gpu:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU 메모리: {total_memory:.2f} GB")
    
    def load_model(self, **kwargs):
        """모델 로딩"""
        if self.model is not None:
            logger.warning("모델이 이미 로드되어 있습니다.")
            self._loaded = True
            return
        
        try:
            logger.info(f"번역 모델 로딩 중: {self.model_path}")
            
            # Tokenizer
            logger.info("Tokenizer 로딩...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # 4bit 양자화
            if self.load_in_4bit and self.use_gpu:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                
                logger.info("4bit 양자화 활성화")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32
                )
            
            self.model.eval()
            self._loaded = True
            
            logger.info("[OK] 번역 모델 로딩 완료")
            logger.info(f"   디바이스: {self.device}")
            logger.info(f"   4bit 양자화: {self.load_in_4bit}")
            
            if self.use_gpu:
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"   모델 메모리: {allocated:.2f} GB")
            
        except Exception as e:
            # Windows 콘솔 호환성을 위해 이모지 제거
            error_msg = f"모델 로딩 실패: {e}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            raise
    
    def translate(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "ja",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
        enable_diarization: bool = True,
        **kwargs
    ) -> TranslationResult:
        """텍스트 번역 (자동 형식 감지)"""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 형식 자동 감지
        format_type = self._detect_format(text)
        
        # 자동 라우팅
        if format_type == "transcript":
            logger.info("[TRANSCRIPT] 자동 감지: 트랜스크립트 형식")
            result_dict = self._translate_transcript(
                text, source_lang, target_lang,
                temperature, top_p, do_sample,
                enable_diarization
            )
        elif format_type == "multiline":
            logger.info("[MULTILINE] 자동 감지: 여러 줄 텍스트")
            result_dict = self._translate_multiline(
                text, source_lang, target_lang,
                max_new_tokens, temperature, top_p, do_sample
            )
        else:
            logger.info("[SINGLE] 자동 감지: 일반 텍스트")
            result_dict = self._translate_single(
                text, source_lang, target_lang,
                max_new_tokens, temperature, top_p, do_sample
            )
        
        # TranslationResult로 변환
        return TranslationResult(
            original_text=result_dict["original_text"],
            translated_text=result_dict["translated_text"],
            source_lang=result_dict["source_lang"],
            target_lang=result_dict["target_lang"],
            input_tokens=result_dict["input_tokens"],
            output_tokens=result_dict["output_tokens"],
            model_name=self.model_name
        )
    
    def _translate_single(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: Optional[int],
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> Dict[str, any]:
        """단일 텍스트 번역"""
        try:
            logger.info(f"번역 시작: {source_lang} → {target_lang}")
            logger.info(f"원문 길이: {len(text)} 글자")
            
            # 프롬프트 생성
            prompt = self._create_prompt(text, source_lang, target_lang)
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_INPUT_LENGTH
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            logger.info(f"입력 토큰 길이: {input_length}")
            
            # max_new_tokens 동적 계산
            if max_new_tokens is None:
                calculated_tokens = int(input_length * 1.5) + 200
                max_new_tokens = min(
                    max(calculated_tokens, self.MIN_OUTPUT_TOKENS),
                    self.MAX_OUTPUT_CAP
                )
                logger.info(f"[OK] 동적 max_new_tokens: {max_new_tokens}")
            else:
                max_new_tokens = min(max_new_tokens, self.MAX_OUTPUT_CAP)
            
            # 메모리 체크 로직은 필요 시 디버깅용으로만 사용 (로그 노이즈 제거를 위해 기본 비활성화)
            # if self.use_gpu:
            #     allocated = torch.cuda.memory_allocated() / 1e9
            #     reserved = torch.cuda.memory_reserved() / 1e9
            #     total = torch.cuda.get_device_properties(0).total_memory / 1e9
            #     free_gb = total - allocated
            #     logger.info(f"VRAM 상태 - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB, 여유: {free_gb:.2f}GB / {total:.2f}GB")
            #     
            #     # 14B 모델(4bit)은 약 8-10GB 필요하므로 3GB 이하면 경고
            #     if free_gb < 3.0:
            #         logger.warning("[WARNING] VRAM 부족 경고! (14B 모델은 최소 8-10GB 필요)")
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            output_length = outputs.shape[1]
            actual_generated = output_length - input_length
            
            # 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            translated_text = self._extract_translation(generated_text, prompt)
            
            logger.info(f"[OK] 번역 완료: {len(translated_text)} 글자")
            
            if self.use_gpu:
                # 14B 모델 사용 후 메모리 정리 강화
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "input_tokens": input_length,
                "output_tokens": actual_generated
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("[ERROR] GPU 메모리 부족!")
                if self.use_gpu:
                    torch.cuda.empty_cache()
                raise RuntimeError("GPU 메모리 부족. 텍스트를 더 짧게 나누세요.") from e
            raise
        except Exception as e:
            logger.error(f"[ERROR] 번역 실패: {e}")
            raise
    
    def _translate_multiline(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: Optional[int],
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> Dict[str, any]:
        """여러 줄 텍스트 번역"""
        lines = text.strip().split('\n')
        translated_lines = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        logger.info(f"줄 단위 번역: {len(lines)}줄")
        
        for i, line in enumerate(lines, 1):
            if line.strip():
                try:
                    result = self._translate_single(
                        text=line.strip(),
                        source_lang=source_lang,
                        target_lang=target_lang,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample
                    )
                    translated_lines.append(result['translated_text'])
                    total_input_tokens += result['input_tokens']
                    total_output_tokens += result['output_tokens']
                    logger.debug(f"  {i}/{len(lines)} 완료")
                except Exception as e:
                    logger.error(f"  {i}번째 줄 실패: {e}")
                    translated_lines.append(f"[번역 실패: {line}]")
            else:
                translated_lines.append('')
        
        logger.info(f"[OK] 줄 단위 번역 완료: {len(lines)}줄")
        
        return {
            "original_text": text,
            "translated_text": '\n'.join(translated_lines),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
    
    def _translate_transcript(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        temperature: float,
        top_p: float,
        do_sample: bool,
        enable_diarization: bool = True
    ) -> Dict[str, any]:
        """트랜스크립트 번역"""
        lines = text.strip().split('\n')
        translated_lines = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        logger.info(f"트랜스크립트 번역: {len(lines)}줄 (화자분리: {'ON' if enable_diarization else 'OFF'})")
        print(f"\n[DEBUG] ===== _translate_transcript 시작 =====")
        print(f"[DEBUG] enable_diarization = {enable_diarization} (type: {type(enable_diarization)})")
        print(f"[DEBUG] 화자분리 모드: {'ON' if enable_diarization else 'OFF'}")
        logger.info(f"원본 텍스트 샘플 (첫 3줄):")
        for i, line in enumerate(lines[:3], 1):
            logger.info(f"  {i}: {line[:100]}")
        
        # 완전히 분리된 처리: True와 False는 독립적인 로직
        if enable_diarization:
            print(f"[DEBUG] ===== 화자분리 ON 블록 실행 =====\n")
            # ===== 화자분리 ON: [타임스탬프] 화자: 내용 형식 =====
            pattern_with_speaker = r'^(\[[^\]]+\])?\s*(화자\d+|Speaker\d+|[^:]+):\s*(.+)$'
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                if not line:
                    translated_lines.append('')
                    continue
                
                try:
                    match = re.match(pattern_with_speaker, line)
                    
                    if match:
                        timestamp = match.group(1) or ''
                        speaker = match.group(2)
                        content = match.group(3)
                        
                        # 내용만 번역
                        text_to_translate = content.strip()
                        
                        # 번역
                        result = self._translate_single(
                            text=text_to_translate,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            max_new_tokens=None,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                        
                        # 번역 결과에서 타임스탬프 형식 제거
                        translated_text = self._remove_timestamps_from_text(result['translated_text'])
                        
                        # 재조립: 타임스탬프와 화자 모두 다시 붙임
                        if timestamp:
                            reconstructed = f"{timestamp} {speaker}: {translated_text}"
                        else:
                            reconstructed = f"{speaker}: {translated_text}"
                        
                        translated_lines.append(reconstructed)
                        total_input_tokens += result['input_tokens']
                        total_output_tokens += result['output_tokens']
                        logger.debug(f"  {i}/{len(lines)} [{speaker}] 완료")
                    else:
                        # 패턴 불일치 - 일반 번역
                        cleaned_line = self._remove_timestamps_from_text(line)
                        if not cleaned_line:
                            translated_lines.append('')
                            continue
                        
                        result = self._translate_single(
                            text=cleaned_line,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            max_new_tokens=None,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                        translated_text = self._remove_timestamps_from_text(result['translated_text'])
                        translated_lines.append(translated_text)
                        total_input_tokens += result['input_tokens']
                        total_output_tokens += result['output_tokens']
                        
                except Exception as e:
                    logger.error(f"  {i}번째 줄 실패: {e}")
                    translated_lines.append(f"[번역 실패] {line}")
        else:
            print(f"[DEBUG] ===== 화자분리 OFF 블록 실행 =====\n")
            # ===== 화자분리 OFF: [타임스탬프] 내용 형식 (화자 정보 없음) =====
            # Whisper 정확한 형식만 매칭: [MM:SS.mmm] 또는 [HH:MM:SS.mmm]
            # Optional(?) 제거: 타임스탬프가 반드시 있어야 함
            # ^\s* 추가: 줄 시작의 공백/BOM 허용
            pattern_without_speaker = r'^\s*(\[\d{2}:\d{2}(?::\d{2})?\.\d{3}\])\s+(.*)$'
            
            print(f"[DEBUG] 화자분리 OFF 모드로 처리 시작 (패턴: {pattern_without_speaker})")
            
            for i, line in enumerate(lines, 1):
                # 원본 라인 보존 (디버깅용)
                original_line = line
                # BOM 제거는 패턴에서 \s*로 처리하므로 여기서는 strip만
                line = line.strip()
                
                if not line:
                    translated_lines.append('')
                    continue
                
                try:
                    print(f"[DEBUG] 줄 {i}: 원본 = '{original_line[:80]}...'")
                    print(f"[DEBUG] 줄 {i}: 정리 후 = '{line[:80]}...'")
                    match = re.match(pattern_without_speaker, line)
                    print(f"[DEBUG] 줄 {i}: 패턴 매칭 결과 = {match is not None}")
                    
                    if match:
                        # 타임스탬프가 정확히 매칭됨
                        timestamp = match.group(1)
                        content = match.group(2)
                        
                        # 디버깅: 원본 타임스탬프 확인 (print로도 출력하여 확실히 확인)
                        print(f"\n[DEBUG] 줄 {i}: 원본 라인 = {line}")
                        print(f"[DEBUG] 줄 {i}: 추출된 타임스탬프 = '{timestamp}'")
                        print(f"[DEBUG] 줄 {i}: 추출된 내용 = '{content[:50]}...'")
                        logger.info(f"  줄 {i}: 원본 라인 = {line}")
                        logger.info(f"  줄 {i}: 추출된 타임스탬프 = '{timestamp}'")
                        logger.info(f"  줄 {i}: 추출된 내용 = '{content[:50]}...'")
                        
                        # 내용만 번역 (타임스탬프는 제외)
                        text_to_translate = content.strip()
                        
                        # 디버깅: LLM에 전달되는 텍스트 확인 (타임스탬프가 포함되지 않았는지 확인)
                        print(f"[DEBUG] 줄 {i}: LLM에 전달할 텍스트 = '{text_to_translate[:80]}...'")
                        logger.info(f"  줄 {i}: LLM에 전달할 텍스트 = '{text_to_translate[:80]}...'")
                        if '[' in text_to_translate and ']' in text_to_translate:
                            print(f"[WARNING] 줄 {i}: LLM에 전달할 텍스트에 타임스탬프 형식이 포함되어 있습니다!")
                            logger.warning(f"  ⚠️ 줄 {i}: LLM에 전달할 텍스트에 타임스탬프 형식이 포함되어 있습니다!")
                        
                        # 번역
                        result = self._translate_single(
                            text=text_to_translate,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            max_new_tokens=None,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                        
                        # 번역 결과에서 타임스탬프 형식 제거 (강화)
                        translated_text = self._remove_timestamps_from_text(result['translated_text'])
                        
                        # 재조립: 원본 타임스탬프를 그대로 사용 (변형하지 않음)
                        reconstructed = f"{timestamp} {translated_text}"
                        logger.info(f"  줄 {i}: 재조립 결과 = '{reconstructed[:80]}...'")
                        
                        translated_lines.append(reconstructed)
                        total_input_tokens += result['input_tokens']
                        total_output_tokens += result['output_tokens']
                        logger.debug(f"  {i}/{len(lines)} 완료")
                    else:
                        # 타임스탬프 패턴 매칭 실패 - 타임스탬프가 없는 줄이거나 형식이 잘못됨
                        logger.warning(f"  줄 {i}: 타임스탬프 패턴 매칭 실패 - '{line[:50]}...'")
                        
                        # 타임스탬프가 없는 줄로 간주하고 전체를 번역
                        # 하지만 먼저 타임스탬프 형식이 있는지 확인
                        if re.search(r'\[\d{1,2}[:：]\d{1,2}', line):
                            # 타임스탬프 형식이 있지만 정확히 매칭되지 않음
                            # 타임스탬프 부분을 제거하고 나머지만 번역
                            cleaned_line = self._remove_timestamps_from_text(line)
                            if cleaned_line and cleaned_line.strip():
                                text_to_translate = cleaned_line.strip()
                            else:
                                # 타임스탬프만 있는 줄
                                translated_lines.append(line)
                                continue
                        else:
                            # 타임스탬프가 없는 일반 텍스트
                            text_to_translate = line
                        
                        # 번역
                        result = self._translate_single(
                            text=text_to_translate,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            max_new_tokens=None,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                        
                        # 번역 결과에서 타임스탬프 형식 제거
                        translated_text = self._remove_timestamps_from_text(result['translated_text'])
                        translated_lines.append(translated_text)
                        total_input_tokens += result['input_tokens']
                        total_output_tokens += result['output_tokens']
                        
                except Exception as e:
                    logger.error(f"  {i}번째 줄 실패: {e}")
                    translated_lines.append(f"[번역 실패] {line}")
        
        logger.info(f"[OK] 트랜스크립트 번역 완료: {len(lines)}줄")
        
        return {
            "original_text": text,
            "translated_text": '\n'.join(translated_lines),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
    
    def _create_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """번역 프롬프트 생성"""
        lang_map = {
            "ko": "Korean",
            "ja": "Japanese", 
            "en": "English"
        }
        
        source_full = lang_map.get(source_lang.lower(), source_lang)
        target_full = lang_map.get(target_lang.lower(), target_lang)
        direction = f"[{source_full} to {target_full}]"
        
        # 강화된 시스템 프롬프트
        system_content = (
            "You are a professional Korean-Japanese bilingual translator. "
            "Translate ONLY the given text accurately without any explanations, "
            "notes, or additional content. Do not mix other languages. "
            "Output only the translated text."
        )
        
        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user", 
                "content": f"{direction}\n{text}"
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        return prompt

    def _remove_timestamps_from_text(self, text: str) -> str:
        """번역 결과에서 타임스탬프 형식 제거 (기계적 처리, LLM 생성 타임스탬프 포함)"""
        if not text:
            return text
        
        result = text
        
        # 1. 정확한 Whisper 형식 제거
        exact_patterns = [
            r'\[\d{2}:\d{2}\.\d{3}\]',  # [00:35.560] 형식
            r'\[\d{2}:\d{2}:\d{2}\.\d{3}\]',  # [00:04:08.000] 형식
        ]
        
        for pattern in exact_patterns:
            result = re.sub(pattern, '', result)
        
        # 2. LLM이 생성한 변형된 타임스탬프 제거 (불완전한 형식 포함)
        weird_patterns = [
            r'\[\d{1,2}:\s*\d{1,2}:\d+\]',  # [00: 00:001] 형식
            r'\[\d{1,2}:\s*\d{1,2}:\d+:\d+\]',  # [00: 21:18:0] 형식
            r'\[\d{1,2}:\s*\d+,\d+초?\]',  # [00: 19,62초] 형식
            r'\[\d{1,2}:\s*\d+:\d+\]',  # [00: 32:82] 형식
            r'\[\d{1,2}:\d{2}(?:\.\d+)?\]',  # [00:35.560] 또는 [00:35] 형식
            r'\[\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\]',  # [00:04:08] 형식
        ]
        
        for pattern in weird_patterns:
            result = re.sub(pattern, '', result)
        
        # 3. 불완전한 타임스탬프 제거 (닫는 괄호가 없는 경우)
        incomplete_patterns = [
            r'\[\d{1,2}:\s*\d{1,2}:\d+[^\]]*$',  # [00: 00:001 (줄 끝)
            r'\[\d{1,2}:\s*\d+[^\]]*$',  # [00: 35 (줄 끝)
            r'\[\d{1,2}:\d{2}[^\]]*$',  # [00:35 (줄 끝)
        ]
        
        for pattern in incomplete_patterns:
            result = re.sub(pattern, '', result, flags=re.MULTILINE)
        
        # 4. 중첩된 대괄호나 이상한 형식 제거
        result = re.sub(r'\[[^\]]*\[', '', result)  # [xxx[ 형식
        result = re.sub(r'\[[^\]]*$', '', result, flags=re.MULTILINE)  # 줄 끝의 불완전한 대괄호
        
        # 5. 기타 모든 [xxx] 형식 제거 (마지막에 실행)
        result = re.sub(r'\[[^\]]+\]', '', result)
        
        # 6. 연속된 공백 정리
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _extract_translation(self, generated_text: str, prompt: str) -> str:
        """생성된 텍스트에서 번역 결과만 추출"""
        result = generated_text
        
        # 1. 프롬프트 제거
        if prompt in result:
            result = result.replace(prompt, "").strip()
        
        # 2. special tokens 제거
        special_tokens = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system\n", "user\n", "assistant\n",
            "<|system|>", "<|user|>", "<|assistant|>"
        ]
        
        for token in special_tokens:
            result = result.replace(token, "")
        
        # 3. 타임스탬프 형식 제거 (번역 결과에 포함될 수 있음)
        result = self._remove_timestamps_from_text(result)
        
        # 4. 설명 패턴 제거
        explanation_patterns = [
            r'[.。]?\s*이\s*문장[^.。]*[.。]',
            r'[.。]?\s*라는\s*의미[^.。]*[.。]',
            r'[.。]?\s*를\s*뜻[^.。]*[.。]',
            r'[.。]?\s*원문의\s*맥락[^.。]*[.。]',
            r'[.。]?\s*자연스럽게\s*표현하면[^.。]*[.。]',
            r'\*\*[^*]+\*\*',
            r'또는\s*\n',
        ]
        
        for pattern in explanation_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        # 5. 다른 언어 섞임 감지
        english_ratio = len(re.findall(r'[a-zA-Z]{3,}', result)) / max(len(result.split()), 1)
        if english_ratio > 0.3:
            logger.warning(f"영어 비율 높음: {english_ratio:.2%}")
        
        # 6. 줄 단위로 라벨 제거
        lines = result.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line.lower() in ['system', 'user', 'assistant']:
                continue
            if len(line) > 0 and not line.replace(' ', '').replace('*', '').replace('-', ''):
                continue
            if line:
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        result = result.strip()
        
        # 7. 최종 검증
        if not result:
            logger.warning("[WARNING] 번역 결과가 비어있습니다!")
            logger.debug(f"생성된 텍스트: {generated_text[:200]}...")
    
        return result
    
    def get_memory_stats(self) -> Dict[str, float]:
        """현재 메모리 사용량 조회"""
        if not self.use_gpu:
            return {"message": "GPU를 사용하지 않습니다."}
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        
        return {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2)
        }
    
    def unload_model(self):
        """메모리 해제 (14B 모델 최적화)"""
        if self.model is not None:
            try:
                logger.info("🔄 Qwen 모델 GPU 메모리 해제 중...")
                
                # GPU에서 CPU로 이동 (GPU 메모리 확보)
                if hasattr(self.model, 'to'):
                    try:
                        self.model.to('cpu')
                    except Exception as e:
                        logger.debug(f"모델 CPU 이동 중 오류 (무시): {e}")
                
                # 모델의 모든 파라미터를 CPU로 명시적으로 이동
                if hasattr(self.model, 'parameters'):
                    for param in self.model.parameters():
                        if param.is_cuda:
                            try:
                                param.data = param.data.cpu()
                            except Exception:
                                pass
                
                # 모델의 모든 버퍼를 CPU로 이동
                if hasattr(self.model, 'buffers'):
                    for buffer in self.model.buffers():
                        if buffer.is_cuda:
                            try:
                                buffer.data = buffer.data.cpu()
                            except Exception:
                                pass
                
                # 4bit 양자화 모델의 특별 처리
                if self.load_in_4bit:
                    try:
                        # BitsAndBytesConfig로 로드된 모델의 특별 처리
                        if hasattr(self.model, 'model'):
                            # PEFT나 양자화 래퍼 제거
                            inner_model = getattr(self.model, 'model', None)
                            if inner_model is not None:
                                if hasattr(inner_model, 'to'):
                                    try:
                                        inner_model.to('cpu')
                                    except Exception:
                                        pass
                    except Exception as e:
                        logger.debug(f"4bit 모델 특별 처리 중 오류 (무시): {e}")
                
                # 모델 삭제
                del self.model
                self.model = None
                
                # Tokenizer도 정리
                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None
                
                # 가비지 컬렉션 실행 (두 번 실행하여 순환 참조 정리)
                import gc
                gc.collect()
                gc.collect()
                
                # GPU 메모리 정리
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                    
                    # 현재 GPU 메모리 사용량 로깅
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    logger.info(f"✅ Qwen 모델 언로드 완료 (GPU 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB)")
                else:
                    logger.info("✅ Qwen 모델 언로드 완료")
                
                self._loaded = False
                
            except Exception as e:
                logger.warning(f"Qwen 모델 언로드 중 오류 (무시): {e}")
                # 오류가 나도 상태는 초기화
                self.model = None
                self.tokenizer = None
                self._loaded = False
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                    

