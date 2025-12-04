#!/usr/bin/env python3
"""
타임스탬프 분리 → 번역 → 문맥 정리 → 타임스탬프 재조립 테스트

프로세스:
1. test/input.txt 읽기
2. 타임스탬프 분리
3. 각 줄 번역 (배열에 저장)
4. 번역된 텍스트들을 모아서 문맥 정리 (전체를 한 번에 LLM에 보내서 자연스럽게 정리)
5. 타임스탬프에 다시 붙이기
6. test/output.txt로 출력
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from api.translation.qwen_local import QwenLocalTranslator
from api.config import TRANSLATION_MODELS
import logging
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_timestamp_line(line: str) -> Tuple[str, str]:
    """
    타임스탬프 라인 파싱
    
    Args:
        line: "[00:00.000] 내용" 형식의 라인
        
    Returns:
        (timestamp, content) 튜플
    """
    line = line.strip()
    if not line:
        return None, None
    
    # 타임스탬프 패턴: [MM:SS.mmm] 또는 [HH:MM:SS.mmm]
    pattern = r'^\s*(\[\d{2}:\d{2}(?::\d{2})?\.\d{3}\])\s+(.*)$'
    match = re.match(pattern, line)
    
    if match:
        timestamp = match.group(1)
        content = match.group(2).strip()
        return timestamp, content
    else:
        # 타임스탬프가 없는 경우
        return None, line


def improve_single_context(translator: QwenLocalTranslator, original_text: str,
                          translated_text: str, source_lang: str, target_lang: str) -> str:
    """
    단일 번역을 원문과 비교하여 문맥에 맞게 자연스럽게 정리
    
    Args:
        translator: QwenLocalTranslator 인스턴스
        original_text: 원문 텍스트
        translated_text: 번역된 텍스트
        source_lang: 원본 언어
        target_lang: 목표 언어
        
    Returns:
        문맥 정리된 번역 텍스트
    """
    if not original_text or not translated_text:
        return translated_text
    
    # 문맥 정리 프롬프트 생성
    lang_map = {
        "ko": "Korean",
        "ja": "Japanese", 
        "en": "English"
    }
    
    source_full = lang_map.get(source_lang.lower(), source_lang)
    target_full = lang_map.get(target_lang.lower(), target_lang)
    
    system_content = (
        f"You are a professional {source_full}-{target_full} bilingual translator. "
        "Compare the original text with its translation and improve the translation to ensure: "
        "1. Accurate word choice and terminology "
        "2. Natural and contextually appropriate phrasing "
        "3. Proper preservation of the original meaning and nuance "
        "Output only the improved translation, nothing else."
    )
    
    user_content = (
        f"Original: {original_text}\n"
        f"Translation: {translated_text}\n\n"
        f"Improve the translation by comparing it with the original text. "
        f"Output only the improved translation."
    )
    
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    prompt = translator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # 토크나이징
    inputs = translator.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=translator.MAX_INPUT_LENGTH
    ).to(translator.device)
    
    input_length = inputs['input_ids'].shape[1]
    max_new_tokens = min(int(input_length * 1.5) + 200, translator.MAX_OUTPUT_CAP)
    
    # 생성
    with torch.no_grad():
        outputs = translator.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=translator.tokenizer.pad_token_id,
            eos_token_id=translator.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    # 디코딩
    generated_text = translator.tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # _extract_translation 메서드 사용
    result_text = translator._extract_translation(generated_text, prompt)
    
    return result_text.strip()


def improve_context(translator: QwenLocalTranslator, original_texts: List[str], 
                   translated_texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """
    번역된 텍스트들을 원문과 비교하여 문맥에 맞게 자연스럽게 정리
    
    Args:
        translator: QwenLocalTranslator 인스턴스
        original_texts: 원문 텍스트 리스트
        translated_texts: 번역된 텍스트 리스트
        source_lang: 원본 언어
        target_lang: 목표 언어
        
    Returns:
        문맥 정리된 번역 텍스트 리스트
    """
    if not translated_texts or not original_texts:
        return translated_texts if translated_texts else []
    
    if len(original_texts) != len(translated_texts):
        logger.warning(f"원문과 번역본 줄 수 불일치: 원문 {len(original_texts)}줄, 번역본 {len(translated_texts)}줄")
        return translated_texts
    
    # 원문과 번역본을 줄 단위로 매칭하여 표시
    comparison_lines = []
    for i, (orig, trans) in enumerate(zip(original_texts, translated_texts), 1):
        comparison_lines.append(f"[{i}] 원문: {orig}")
        comparison_lines.append(f"[{i}] 번역: {trans}")
        comparison_lines.append("")  # 빈 줄로 구분
    
    comparison_text = '\n'.join(comparison_lines)
    
    logger.info(f"문맥 정리 시작: {len(translated_texts)}개 문장")
    logger.info(f"원문과 번역본 비교 텍스트 길이: {len(comparison_text)} 글자")
    
    # 문맥 정리 프롬프트 생성
    lang_map = {
        "ko": "Korean",
        "ja": "Japanese", 
        "en": "English"
    }
    
    source_full = lang_map.get(source_lang.lower(), source_lang)
    target_full = lang_map.get(target_lang.lower(), target_lang)
    
    system_content = (
        f"You are a professional {source_full}-{target_full} bilingual translator. "
        "You will be given the original text and its translation in pairs. "
        "Please improve each translation by comparing it with the original text to ensure: "
        "1. Accurate word choice and terminology "
        "2. Natural and contextually appropriate phrasing "
        "3. Proper preservation of the original meaning and nuance "
        "4. Consistent style throughout the text "
        "\n"
        "CRITICAL OUTPUT FORMAT: "
        "You must output EXACTLY one improved translation per line. "
        "Each line must be separated by a newline character (\\n). "
        "Do NOT combine multiple translations into one line. "
        "Do NOT add any line numbers, prefixes, or explanations. "
        f"Output exactly {len(translated_texts)} lines, one translation per line."
    )
    
    user_content = (
        f"Compare the following {len(translated_texts)} pairs of original text and translation, "
        f"then improve each translation to be more accurate and natural.\n\n"
        f"{comparison_text}\n\n"
        f"OUTPUT FORMAT: Write each improved translation on a separate line. "
        f"Line 1: improved translation for pair [1]\n"
        f"Line 2: improved translation for pair [2]\n"
        f"... and so on.\n"
        f"Output exactly {len(translated_texts)} lines, one per line, in order."
    )
    
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    prompt = translator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # 토크나이징
    inputs = translator.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=translator.MAX_INPUT_LENGTH
    ).to(translator.device)
    
    input_length = inputs['input_ids'].shape[1]
    max_new_tokens = min(int(input_length * 1.5) + 200, translator.MAX_OUTPUT_CAP)
    
    # 생성
    with torch.no_grad():
        outputs = translator.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,  # 문맥 정리는 더 낮은 temperature 사용
            top_p=0.9,
            do_sample=True,
            pad_token_id=translator.tokenizer.pad_token_id,
            eos_token_id=translator.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    # 디코딩
    generated_text = translator.tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # _extract_translation 메서드 사용 (기존 로직 재사용)
    result_text = translator._extract_translation(generated_text, prompt)
    
    logger.info(f"문맥 정리 원시 결과 (처음 200자): {result_text[:200]}...")
    
    # 줄 단위로 분리 (더 정확한 파싱)
    # 먼저 줄바꿈으로 분리
    lines = result_text.split('\n')
    improved_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 줄 번호나 특수 문자 제거 ([1], [2] 등)
        line = re.sub(r'^\[\d+\]\s*', '', line)
        line = re.sub(r'^\d+\.\s*', '', line)
        line = line.strip()
        
        if line:
            improved_lines.append(line)
    
    logger.info(f"파싱된 줄 수: {len(improved_lines)}줄 (목표: {len(translated_texts)}줄)")
    
    # 원본 줄 수와 맞추기
    if len(improved_lines) != len(translated_texts):
        logger.warning(
            f"문맥 정리 결과 줄 수 불일치: 원본 {len(translated_texts)}줄, "
            f"결과 {len(improved_lines)}줄"
        )
        
        # 결과가 1줄이면 문장 단위로 분리 시도
        if len(improved_lines) == 1 and len(translated_texts) > 1:
            logger.info("결과가 1줄로 반환됨. 문장 단위로 분리 시도...")
            single_line = improved_lines[0]
            
            # 더 정확한 문장 분리: 문장 종결 기호 + 공백/줄바꿈
            # 한국어 문장 종결: . ! ? 。 ！ ？
            # 문장 종결 기호 뒤에 공백이나 줄바꿈이 오는 경우 분리
            pattern = r'([.!?。！？])\s+'
            sentences = re.split(pattern, single_line)
            
            # 분리된 문장 재조립
            separated = []
            current_sentence = ""
            for i, part in enumerate(sentences):
                if part in '.!?。！？':
                    current_sentence += part
                    if current_sentence.strip():
                        separated.append(current_sentence.strip())
                    current_sentence = ""
                else:
                    current_sentence += part
            
            # 마지막 문장 처리
            if current_sentence.strip():
                separated.append(current_sentence.strip())
            
            logger.info(f"문장 분리 결과: {len(separated)}개 문장 (목표: {len(translated_texts)}줄)")
            
            # 원본 번역본의 길이를 참고하여 더 정확하게 분리
            if len(separated) != len(translated_texts):
                # 원본 번역본의 평균 길이를 기준으로 분리 시도
                avg_length = sum(len(t) for t in translated_texts) / len(translated_texts)
                logger.info(f"원본 번역본 평균 길이: {avg_length:.1f} 글자")
                
                # 문장을 길이 기준으로 재분리 시도
                if len(separated) < len(translated_texts):
                    # 문장이 적으면 더 세밀하게 분리
                    new_separated = []
                    for sent in separated:
                        # 문장이 너무 길면 쉼표나 연결어로 분리
                        if len(sent) > avg_length * 1.5:
                            parts = re.split(r'([,，]\s*)', sent)
                            temp = ""
                            for part in parts:
                                temp += part
                                if len(temp) >= avg_length * 0.8:
                                    new_separated.append(temp.strip())
                                    temp = ""
                            if temp.strip():
                                new_separated.append(temp.strip())
                        else:
                            new_separated.append(sent)
                    separated = new_separated
            
            if len(separated) == len(translated_texts):
                improved_lines = separated
                logger.info(f"문장 분리 성공: {len(improved_lines)}줄")
            elif len(separated) > len(translated_texts):
                # 너무 많이 분리된 경우 앞에서부터 자르기
                improved_lines = separated[:len(translated_texts)]
                logger.warning(f"문장 분리 초과 - 앞 {len(translated_texts)}개만 사용")
            else:
                logger.warning(f"문장 분리 실패: {len(separated)}개 문장 (목표: {len(translated_texts)}줄)")
        
        # 줄 수가 다르면 원본 번역 사용
        if len(improved_lines) < len(translated_texts):
            logger.warning("줄 수 부족 - 원본 번역으로 보완")
            improved_lines.extend(translated_texts[len(improved_lines):])
        elif len(improved_lines) > len(translated_texts):
            logger.warning("줄 수 초과 - 초과분 제거")
            improved_lines = improved_lines[:len(translated_texts)]
    
    logger.info(f"문맥 정리 완료: {len(improved_lines)}개 문장")
    
    return improved_lines


def main():
    """메인 함수"""
    
    # 경로 설정
    test_dir = project_root / "test"
    input_file = test_dir / "input.txt"
    output_file = test_dir / "output.txt"
    
    # 입력 파일 확인
    if not input_file.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_file}")
        return
    
    # 입력 파일 읽기
    logger.info(f"입력 파일 읽기: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logger.info(f"총 {len(lines)}줄 읽음")
    
    # 타임스탬프와 내용 분리
    timestamp_content_pairs = []
    for line in lines:
        timestamp, content = parse_timestamp_line(line)
        if timestamp and content:
            timestamp_content_pairs.append((timestamp, content))
        elif content:
            # 타임스탬프가 없는 경우
            timestamp_content_pairs.append((None, content))
    
    logger.info(f"파싱 완료: {len(timestamp_content_pairs)}개 항목")
    
    # 모델 경로 설정 (qwen3-14b-lora-10ratio 명시적 사용)
    base_model_dir = project_root / "qwen3-14b-lora-10ratio"
    
    # LoRA 모델의 경우 하위 폴더 확인 (우선순위 순서)
    possible_paths = [
        base_model_dir / "qwen3-14b-lora-10ratio",  # 하위 폴더 (실제 모델 위치)
        base_model_dir,  # 루트 폴더
        base_model_dir / "checkpoint-8068",  # 최신 체크포인트
    ]
    
    actual_model_path = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            # config.json이나 tokenizer.json이 있는지 확인
            if (path / "config.json").exists() or (path / "tokenizer.json").exists():
                actual_model_path = path
                logger.info(f"모델 경로 찾음: {actual_model_path}")
                break
    
    if actual_model_path is None:
        logger.error(f"모델 경로를 찾을 수 없습니다. 시도한 경로:")
        for path in possible_paths:
            logger.error(f"  - {path} (존재: {path.exists() if path else False})")
        logger.info("\n사용 가능한 모델:")
        for key, path in TRANSLATION_MODELS.items():
            logger.info(f"  {key}: {path} (존재: {path.exists() if path else False})")
        return
    
    model_path = actual_model_path
    
    # 번역기 초기화
    logger.info(f"번역 모델 로딩: {model_path}")
    translator = QwenLocalTranslator(
        model_path=str(model_path),
        use_gpu=torch.cuda.is_available(),
        load_in_4bit=True
    )
    
    try:
        # 모델 로드
        translator.load_model()
        
        # 언어 설정 (입력 파일이 일본어인 것으로 가정)
        source_lang = "ja"
        target_lang = "ko"
        
        # 1단계: 각 줄 번역
        logger.info("=" * 60)
        logger.info("1단계: 각 줄 번역 시작")
        logger.info("=" * 60)
        
        original_texts = []  # 원문 저장
        translated_texts = []
        for i, (timestamp, content) in enumerate(timestamp_content_pairs, 1):
            logger.info(f"[{i}/{len(timestamp_content_pairs)}] 번역 중: {content[:50]}...")
            
            # 원문 저장
            original_texts.append(content)
            
            result = translator._translate_single(
                text=content,
                source_lang=source_lang,
                target_lang=target_lang,
                max_new_tokens=None,
                temperature=0.1,
                top_p=0.9,
                do_sample=True
            )
            
            # 타임스탬프 제거
            translated_text = translator._remove_timestamps_from_text(result['translated_text'])
            translated_texts.append(translated_text)
            
            logger.info(f"  → {translated_text[:50]}...")
        
        logger.info(f"번역 완료: {len(translated_texts)}개 문장")
        
        # 2단계: 문맥 정리 (원문과 번역본 비교)
        logger.info("=" * 60)
        logger.info("2단계: 문맥 정리 시작 (원문과 번역본 비교)")
        logger.info("=" * 60)
        
        # 전체를 한 번에 처리하는 대신, 각 줄을 개별적으로 처리
        # 이렇게 하면 줄 수 문제를 완전히 해결할 수 있음
        improved_texts = []
        for i, (orig, trans) in enumerate(zip(original_texts, translated_texts), 1):
            logger.info(f"[{i}/{len(translated_texts)}] 문맥 정리 중...")
            improved = improve_single_context(
                translator, orig, trans, source_lang, target_lang
            )
            improved_texts.append(improved)
            logger.info(f"  원문: {orig[:50]}...")
            logger.info(f"  개선: {improved[:50]}...")
        
        # 3단계: 타임스탬프 재조립
        logger.info("=" * 60)
        logger.info("3단계: 타임스탬프 재조립")
        logger.info("=" * 60)
        
        output_lines = []
        for (timestamp, _), improved_text in zip(timestamp_content_pairs, improved_texts):
            if timestamp:
                output_lines.append(f"{timestamp} {improved_text}")
            else:
                output_lines.append(improved_text)
        
        # 출력 파일 저장
        logger.info(f"출력 파일 저장: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            f.write('\n')
        
        logger.info("=" * 60)
        logger.info("처리 완료!")
        logger.info("=" * 60)
        logger.info(f"입력: {input_file}")
        logger.info(f"출력: {output_file}")
        logger.info(f"총 {len(output_lines)}줄 처리됨")
        
        # 비교 출력
        logger.info("\n" + "=" * 60)
        logger.info("번역 결과 비교 (처음 3줄):")
        logger.info("=" * 60)
        for i in range(min(3, len(translated_texts))):
            logger.info(f"\n원본 번역 [{i+1}]:")
            logger.info(f"  {translated_texts[i]}")
            logger.info(f"\n문맥 정리 [{i+1}]:")
            logger.info(f"  {improved_texts[i]}")
        
    finally:
        # 모델 언로드
        translator.unload_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

