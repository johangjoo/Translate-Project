"""
Qwen3-8b LoRA 번역 모듈 (API 통합 버전)
위치: api/translation.py
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
    """Qwen3-8b LoRA 기반 번역 모델"""
    
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
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def translate(
        self,
        text: str,
        source_lang: str = "ko",
        target_lang: str = "ja",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, str]:
        """
        텍스트 번역
        
        Args:
            text: 번역할 텍스트
            source_lang: 원본 언어 (ko, ja, en)
            target_lang: 목표 언어 (ko, ja, en)
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: nucleus sampling
            do_sample: 샘플링 사용 여부
        
        Returns:
            {
                "original_text": str,
                "translated_text": str,
                "source_lang": str,
                "target_lang": str
            }
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        try:
            logger.info(f"번역 시작: {source_lang} → {target_lang}")
            logger.info(f"원문: {text[:100]}...")
            
            # 프롬프트 생성
            prompt = self._create_prompt(text, source_lang, target_lang)
            logger.debug(f"생성된 프롬프트:\n{prompt}")
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
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
                    repetition_penalty=1.1,  # 반복 방지
                    no_repeat_ngram_size=3   # n-gram 반복 방지
                )
            
            # 디코딩 - skip_special_tokens=False로 해서 수동 제거
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=False
            )
            
            logger.debug(f"생성된 전체 텍스트:\n{generated_text}")
            
            # 번역 결과 추출
            translated_text = self._extract_translation(generated_text, prompt)
            
            logger.info(f"✅ 번역 완료: {translated_text[:100]}...")
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            
        except Exception as e:
            logger.error(f"❌ 번역 실패: {e}")
            raise
    
    def _create_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """번역 프롬프트 생성 (학습 데이터와 정확히 동일한 형식)"""
        
        # 언어 코드 → 풀네임 변환 (학습 데이터와 일치!)
        lang_map = {
            "ko": "Korean",
            "ja": "Japanese", 
            "en": "English"
        }
        
        source_full = lang_map.get(source_lang.lower(), source_lang)
        target_full = lang_map.get(target_lang.lower(), target_lang)
        
        # 학습 데이터와 동일한 형식: [Japanese to Korean]
        direction = f"[{source_full} to {target_full}]"
        
        messages = [
            {
                "role": "system",
                "content": " You are a professional Korean-Japanese bilingual translator."
            },
            {
                "role": "user", 
                "content": f"{direction}\n{text}"
            }
        ]
        
        # Qwen3 템플릿 적용 - thinking 비활성화 필수!
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 중요!
        )
        
        return prompt

    def _extract_translation(self, generated_text: str, prompt: str) -> str:
        """생성된 텍스트에서 번역 결과만 추출"""
        
        result = generated_text
        
        # 1. 입력 프롬프트 전체 제거
        if prompt in result:
            result = result.replace(prompt, "").strip()
        
        # 2. special tokens 제거 (디코딩 후에도 남아있을 수 있음)
        special_tokens = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system\n", "user\n", "assistant\n",
            "<|system|>", "<|user|>", "<|assistant|>"
        ]
        
        for token in special_tokens:
            result = result.replace(token, "")
        
        # 3. thinking 태그 제거 (<think>...</think>)
        if "<think>" in result:
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        
        # 4. system/user/assistant 라벨 제거 (줄 단위)
        lines = result.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # system, user, assistant로만 된 줄 제거
            if line.lower() in ['system', 'user', 'assistant']:
                continue
            if line:
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        # 5. 앞뒤 공백/줄바꿈 정리
        result = result.strip()
        
        # 6. 번역 결과가 비정상적으로 짧거나 비어있으면 경고
        if not result or len(result) < 5:
            logger.warning(f"번역 결과가 비정상적으로 짧습니다: '{result}'")
            logger.warning(f"생성된 전체 텍스트: {generated_text}")
        
        return result
    
    def unload_model(self):
        """메모리 해제"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            logger.info("번역 모델 언로드 완료")


# 전역 인스턴스
qwen3_translator: Optional[Qwen3Translator] = None


def initialize_translator(
    model_path: str = "qwen3-8b-lora-10ratio",
    use_gpu: bool = True,
    load_in_4bit: bool = True
):
    """
    번역 모델 초기화
    
    Args:
        model_path: LoRA 모델 경로
        use_gpu: GPU 사용 여부
        load_in_4bit: 4bit 양자화 사용 여부
    """
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