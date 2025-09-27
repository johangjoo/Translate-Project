import ollama
import os

# 사용할 Ollama 모델 이름
MODEL_NAME = 'mannix/llamax3-8b-alpaca:q8_0'

def translate_text(text, source_lang, target_lang):
    """
    주어진 텍스트를 지정된 언어로 번역합니다.

    :param text: 번역할 원본 텍스트
    :param source_lang: 원본 언어 (예: Korean)
    :param target_lang: 번역할 대상 언어 (예: English)
    :return: 번역된 텍스트
    """
    print(f"--- {source_lang} -> {target_lang} 번역 시작 ---")
    try:
        # 프롬프트 엔지니어링: LLM에게 명확한 역할과 지시를 내립니다.
        prompt = f"""
        You are a professional translator.
        Translate the following {source_lang} text to {target_lang}.
        Provide only the translated text, without any additional explanations or introductory phrases.

        Original Text:
        "{text}"

        Translated Text:
        """

        # Ollama 서버에 API 요청 보내기
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            options={
                'temperature': 0.3 # 번역의 일관성을 위해 온도를 낮게 설정
            }
        )

        # 응답에서 번역된 텍스트만 추출
        translated_text = response['message']['content'].strip()
        print(translated_text)
        return translated_text

    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    input_file = "test.txt"

    if not os.path.exists(input_file):
        print(f"'{input_file}'을 찾을 수 없습니다. 파일을 생성하고 내용을 입력해주세요.")
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        print(f"원본 텍스트:\n{original_text}\n")

        # 한 -> 영 번역
        translate_text(original_text, "Korean", "English")

        print("\n" + "="*50 + "\n")

        # 한 -> 일 번역
        translate_text(original_text, "Korean", "Japanese")