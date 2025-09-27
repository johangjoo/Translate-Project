import os
from llama_cpp import Llama

# --- 1. 구성 설정 ---
# 다운로드한 Qwen2 모델의 GGUF 파일 경로를 지정합니다.
# 이 부분만 llama3Test.py와 다릅니다.
MODEL_PATH = "./models/qwen2-7b-instruct.Q8_0.gguf" 
INPUT_DIR = "input"
# 출력 디렉토리 이름도 Qwen2에 맞게 변경합니다.
OUTPUT_DIR = "output_qwen2" 
TARGET_LANGUAGES = ["Korean", "English", "Japanese"]

# 모델의 컨텍스트 창 크기. 4096은 7B 모델에 적합한 크기입니다.
CONTEXT_SIZE = 4096 
# 생성할 최대 토큰 수. -1은 컨텍스트가 꽉 찰 때까지 생성을 의미합니다.
MAX_TOKENS = -1
# 번역의 창의성을 조절하는 값. 0.7은 자연스러움과 정확성 사이의 균형을 맞춥니다.
TEMPERATURE = 0.7

# --- 2. 프롬프트 템플릿 ---
# 모든 번역 작업에 일관되게 사용될 페르소나 기반 프롬프트입니다.
PROMPT_TEMPLATE = """You are an expert multilingual translator and localization specialist. Your expertise lies in translating conversational dialogue between Korean, English, and Japanese for scripts and subtitles.

Your task is to translate the provided text from {source_language} to {target_language}.

You must adhere to the following rules:
1. Preserve the original conversational format EXACTLY, including any prefixes like '화자 1:', 'Speaker 2:', etc.
2. Translate the content of the dialogue naturally and accurately, maintaining the original tone and intent of each speaker.
3. Do not add any commentary, explanations, or text outside of the translated dialogue itself. Your output must ONLY be the translated conversation.

---SOURCE TEXT---
{text_to_translate}
---END SOURCE TEXT---

---TRANSLATED TEXT ({target_language})---
"""

def create_output_directories():
    """
    결과물이 저장될 출력 디렉토리를 생성합니다.
    os.makedirs를 exist_ok=True와 함께 사용하여 스크립트를 여러 번 실행해도 오류가 발생하지 않도록 합니다.
    """
    print(f"'{OUTPUT_DIR}' 디렉토리를 생성하거나 확인합니다...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("디렉토리 준비 완료.")

def load_llm_model(model_path):
    """
    지정된 경로의 GGUF 모델을 로드하고 GPU 가속을 설정합니다.
    n_gpu_layers=-1은 가능한 모든 레이어를 GPU로 오프로드하여 최대 성능을 내도록 합니다.
    """
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다 - {model_path}")
        print("스크립트 상단의 MODEL_PATH 변수가 올바른지 확인하거나 모델을 다운로드하세요.")
        return None

    print(f"'{model_path}'에서 모델을 로딩합니다. 이 과정은 몇 분 정도 소요될 수 있습니다...")
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # 모든 레이어를 GPU로 오프로드
            n_ctx=CONTEXT_SIZE,
            verbose=True,     # 로딩 과정에서 상세 정보 출력 (GPU 오프로드 확인에 유용)
        )
        print("모델 로딩 완료.")
        return llm
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        print("llama-cpp-python이 CUDA 지원으로 올바르게 컴파일되었는지 확인하세요.")
        return None


def translate_text(llm, text_content, source_lang, target_lang):
    """
    주어진 텍스트를 지정된 언어로 번역합니다.
    """
    print(f"'{source_lang}'에서 '{target_lang}'(으)로 번역을 시작합니다...")
    
    prompt = PROMPT_TEMPLATE.format(
        source_language=source_lang,
        target_language=target_lang,
        text_to_translate=text_content
    )
    
    try:
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop= # 모델이 불필요한 텍스트를 생성하는 것을 방지
        )
        
        translated_text = output['choices']['text'].strip()
        print("번역 완료.")
        return translated_text
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return None

def main():
    """
    메인 실행 함수: 디렉토리 생성, 모델 로드, 파일 처리, 번역 및 저장을 조율합니다.
    """
    print("--- Qwen2 번역 테스트 시작 ---")
    create_output_directories()
    
    llm = load_llm_model(MODEL_PATH)
    if llm is None:
        return # 모델 로딩 실패 시 종료

    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"오류: '{INPUT_DIR}' 폴더가 비어있거나 존재하지 않습니다.")
        print("번역할.txt 파일을 해당 폴더에 넣어주세요.")
        return

    print(f"'{INPUT_DIR}' 폴더에서 파일을 처리합니다...")
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(INPUT_DIR, filename)
            print(f"\n파일 처리 중: {filename}")

            # 파일 이름에서 소스 언어 추론 (예: ko_dialogue.txt -> Korean)
            source_language = "Unknown"
            if filename.lower().startswith("ko"):
                source_language = "Korean"
            elif filename.lower().startswith("en"):
                source_language = "English"
            elif filename.lower().startswith("ja") or filename.lower().startswith("jp"):
                source_language = "Japanese"
            
            if source_language == "Unknown":
                print(f"경고: '{filename}'의 소스 언어를 식별할 수 없습니다. 건너뜁니다.")
                continue

            try:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"파일 읽기 오류 '{input_filepath}': {e}")
                continue

            for target_lang in TARGET_LANGUAGES:
                # 자기 자신으로의 번역은 건너뜀
                if source_language == target_lang:
                    continue

                translated_content = translate_text(llm, content, source_language, target_lang)
                
                if translated_content:
                    output_filename = f"{os.path.splitext(filename)}_to_{target_lang.lower()}.txt"
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    
                    try:
                        with open(output_filepath, 'w', encoding='utf-8') as f:
                            f.write(translated_content)
                        print(f"결과 저장 완료: {output_filepath}")
                    except Exception as e:
                        print(f"파일 쓰기 오류 '{output_filepath}': {e}")

    print("\n--- 모든 작업 완료 ---")

if __name__ == "__main__":
    main()