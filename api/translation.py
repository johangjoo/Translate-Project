"""
Qwen3-8b LoRA 번역 모듈 (통합 버전)
위치: api/translation.py

자동 형식 감지:
- 트랜스크립트: [mm:ss] 화자: 내용
- 여러 줄 텍스트: 가사, 시 등
- 일반 텍스트: 단순 문장

12GB GPU (RTX 5070 Ti) 최적화
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from pathlib import Path
from typing import Optional, Dict
import re

logger = logging.getLogger(__name__)


class Qwen3Translator:
    """Qwen3-8b LoRA 기반 번역 모델 (자동 형식 감지)"""
    
    # 12GB GPU 최적화 설정
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
        초기화
        
        Args:
            model_path: LoRA 모델 경로
            use_gpu: GPU 사용 여부
            load_in_4bit: 4bit 양자화 사용 여부
        """
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Qwen3Translator 초기화 - 디바이스: {self.device}")
        if self.use_gpu:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU 메모리: {total_memory:.2f} GB")
    
    def load_model(self):
        """모델 로딩"""
        if self.model is not None:
            logger.warning("모델이 이미 로드되어 있습니다.")
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
            
            logger.info("✅ 번역 모델 로딩 완료")
            logger.info(f"   디바이스: {self.device}")
            logger.info(f"   4bit 양자화: {self.load_in_4bit}")
            
            if self.use_gpu:
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"   모델 메모리: {allocated:.2f} GB")
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def translate(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "ja",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.3,  #1일수록 자연스럽고 0일수록 
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, str]:
        """
        텍스트 번역 (🎯 자동 형식 감지)
        
        자동으로 다음 형식을 감지하고 처리:
        - 트랜스크립트: [mm:ss] 화자: 내용 → 타임스탬프+화자 보존
        - 여러 줄 텍스트: 가사, 시 등 → 줄 단위 번역
        - 일반 텍스트: 단순 문장 → 일반 번역
        
        Args:
            text: 번역할 텍스트
            source_lang: 원본 언어 (ko, ja, en)
            target_lang: 목표 언어 (ko, ja, en)
            max_new_tokens: 최대 생성 토큰 수 (None이면 자동 계산)
            temperature: 샘플링 온도
            top_p: nucleus sampling
            do_sample: 샘플링 사용 여부
        
        Returns:
            {
                "original_text": str,
                "translated_text": str,
                "source_lang": str,
                "target_lang": str,
                "input_tokens": int,
                "output_tokens": int
            }
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 형식 자동 감지
        lines = text.strip().split('\n')
        
        # 트랜스크립트 형식 감지: [타임스탬프] 화자: 내용
        transcript_pattern = r'^(\[[\d:\.]+\])?\s*(화자\d+|Speaker\d+|[^:]+):\s*.+$'
        first_line = lines[0].strip() if lines else ""
        is_transcript = bool(re.match(transcript_pattern, first_line))
        
        # 여러 줄 텍스트 (3줄 이상)
        is_multiline = len(lines) >= 3
        
        # 자동 라우팅
        if is_transcript:
            logger.info("📋 자동 감지: 트랜스크립트 형식")
            return self._translate_transcript(
                text, source_lang, target_lang,
                temperature, top_p, do_sample
            )
        elif is_multiline:
            logger.info("📝 자동 감지: 여러 줄 텍스트")
            return self._translate_multiline(
                text, source_lang, target_lang,
                max_new_tokens, temperature, top_p, do_sample
            )
        else:
            logger.info("💬 자동 감지: 일반 텍스트")
            return self._translate_single(
                text, source_lang, target_lang,
                max_new_tokens, temperature, top_p, do_sample
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
    ) -> Dict[str, str]:
        """단일 텍스트 번역 (내부 메서드)"""
        
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
                logger.info(f"✅ 동적 max_new_tokens: {max_new_tokens}")
            else:
                max_new_tokens = min(max_new_tokens, self.MAX_OUTPUT_CAP)
            
            # 메모리 체크
            if self.use_gpu:
                free_memory = (
                    torch.cuda.get_device_properties(0).total_memory 
                    - torch.cuda.memory_allocated()
                )
                free_gb = free_memory / 1e9
                logger.info(f"여유 VRAM: {free_gb:.2f} GB")
                
                if free_gb < 2.0:
                    logger.warning("⚠️ VRAM 부족 경고!")
            
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
            
            logger.info(f"✅ 번역 완료: {len(translated_text)} 글자")
            
            if self.use_gpu:
                torch.cuda.empty_cache()
            
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
                logger.error("❌ GPU 메모리 부족!")
                if self.use_gpu:
                    torch.cuda.empty_cache()
                raise RuntimeError("GPU 메모리 부족. 텍스트를 더 짧게 나누세요.") from e
            raise
        except Exception as e:
            logger.error(f"❌ 번역 실패: {e}")
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
    ) -> Dict[str, str]:
        """여러 줄 텍스트 번역 (내부 메서드)"""
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
        
        logger.info(f"✅ 줄 단위 번역 완료: {len(lines)}줄")
        
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
        do_sample: bool
    ) -> Dict[str, str]:
        """트랜스크립트 번역 (내부 메서드)"""
        lines = text.strip().split('\n')
        translated_lines = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 정규표현식 패턴: [타임스탬프] 화자: 내용
        pattern = r'^(\[[\d:\.]+\])?\s*(화자\d+|Speaker\d+|[^:]+):\s*(.+)$'
        
        logger.info(f"트랜스크립트 번역: {len(lines)}줄")
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                translated_lines.append('')
                continue
            
            match = re.match(pattern, line)
            
            if match:
                timestamp = match.group(1) or ''
                speaker = match.group(2)
                content = match.group(3)
                
                try:
                    # 내용만 번역
                    result = self._translate_single(
                        text=content.strip(),
                        source_lang=source_lang,
                        target_lang=target_lang,
                        max_new_tokens=None,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample
                    )
                    
                    # 재조립
                    if timestamp:
                        reconstructed = f"{timestamp} {speaker}: {result['translated_text']}"
                    else:
                        reconstructed = f"{speaker}: {result['translated_text']}"
                    
                    translated_lines.append(reconstructed)
                    total_input_tokens += result['input_tokens']
                    total_output_tokens += result['output_tokens']
                    logger.debug(f"  {i}/{len(lines)} [{speaker}] 완료")
                    
                except Exception as e:
                    logger.error(f"  {i}번째 줄 실패: {e}")
                    translated_lines.append(f"[번역 실패] {line}")
            else:
                # 패턴 불일치 - 일반 번역
                try:
                    result = self._translate_single(
                        text=line,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        max_new_tokens=None,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample
                    )
                    translated_lines.append(result['translated_text'])
                    total_input_tokens += result['input_tokens']
                    total_output_tokens += result['output_tokens']
                except Exception as e:
                    translated_lines.append(f"[번역 실패] {line}")
        
        logger.info(f"✅ 트랜스크립트 번역 완료: {len(lines)}줄")
        
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
        
        # 3. thinking 태그 제거
        if "<think>" in result:
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        
        # 4. 설명 패턴 제거 (예: "...라는 의미입니다", "...를 뜻합니다")
        explanation_patterns = [
            r'[.。]?\s*이\s*문장[^.。]*[.。]',
            r'[.。]?\s*라는\s*의미[^.。]*[.。]',
            r'[.。]?\s*를\s*뜻[^.。]*[.。]',
            r'[.。]?\s*원문의\s*맥락[^.。]*[.。]',
            r'[.。]?\s*자연스럽게\s*표현하면[^.。]*[.。]',
            r'\*\*[^*]+\*\*',  # **굵은 글씨** 제거
            r'또는\s*\n',       # "또는" 뒤의 추가 설명 제거
        ]
        
        for pattern in explanation_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        # 5. 다른 언어 섞임 감지 및 제거 (영어 단어 덩어리, 일본어 히라가나)
        # 영어 단어가 많이 섞였으면 문제
        english_ratio = len(re.findall(r'[a-zA-Z]{3,}', result)) / max(len(result.split()), 1)
        if english_ratio > 0.3:  # 30% 이상 영어면 의심
            logger.warning(f"영어 비율 높음: {english_ratio:.2%}")
        
        # 6. 줄 단위로 라벨 제거
        lines = result.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 시스템 라벨 제거
            if line.lower() in ['system', 'user', 'assistant']:
                continue
            # 너무 짧거나 특수문자만 있으면 제거
            if len(line) > 0 and not line.replace(' ', '').replace('*', '').replace('-', ''):
                continue
            if line:
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        result = result.strip()
        
        # 7. 최종 검증
        if not result:
            logger.warning("⚠️ 번역 결과가 비어있습니다!")
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
        """메모리 해제"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            if self.use_gpu:
                torch.cuda.empty_cache()
                logger.info("GPU 메모리 정리 완료")
            
            logger.info("번역 모델 언로드 완료")


# 전역 인스턴스
qwen3_translator: Optional[Qwen3Translator] = None


def initialize_translator(
    model_path: str = "qwen3-8b-lora-10ratio",
    use_gpu: bool = True,
    load_in_4bit: bool = True
):
    """번역 모델 초기화"""
    global qwen3_translator
    
    logger.info("="*50)
    logger.info("🚀 번역 모델 초기화 시작...")
    logger.info("="*50)
    
    qwen3_translator = Qwen3Translator(
        model_path=model_path,
        use_gpu=use_gpu,
        load_in_4bit=load_in_4bit
    )
    qwen3_translator.load_model()
    
    logger.info("="*50)
    logger.info("✅ 번역 모델 초기화 완료!")
    logger.info("="*50)


def get_memory_info() -> Dict[str, float]:
    """현재 메모리 정보 조회"""
    if qwen3_translator is None:
        return {"error": "모델이 초기화되지 않았습니다."}
    return qwen3_translator.get_memory_stats()