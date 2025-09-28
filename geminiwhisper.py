import os
import glob
import torch
import whisperx
from moviepy import VideoFileClip
import datetime
import gc

# --- 1. 초기화 및 환경설정 ---

# 입출력 디렉터리 정의
VIDEO_INPUT_DIR = "video_input"
TRANSCRIPT_OUTPUT_DIR = "input"

# WhisperX 모델 설정
# 사용 가능한 GPU가 있으면 'cuda', 없으면 'cpu'를 자동으로 사용합니다.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 모델 크기: 'tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3' 중 선택
# 더 큰 모델은 정확도가 높지만 더 많은 VRAM과 처리 시간을 요구합니다.
MODEL_SIZE = "large-v3"
# GPU 메모리가 부족할 경우 배치 크기를 줄이세요 (예: 8, 4, 2).
BATCH_SIZE = 16
# 최신 GPU에서는 'float16'이 효율적입니다. VRAM이 부족하면 'int8'을 사용해볼 수 있습니다.
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

# Hugging Face 액세스 토큰 (화자 분리 기능에 필요)
# 보안을 위해 실제 토큰은 환경 변수에서 가져옵니다.
# 실행 전 터미널에서 `export HF_TOKEN=your_token_here` (macOS/Linux) 또는
# `set HF_TOKEN=your_token_here` (Windows)를 실행하세요.
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print("경고: Hugging Face 토큰(HF_TOKEN)이 설정되지 않았습니다. 화자 분리 기능이 작동하지 않을 수 있습니다.")

# --- 2. 오디오 추출 모듈 ---

def extract_audio(video_path, audio_path):
    """
    moviepy를 사용하여 비디오 파일에서 오디오를 추출하고 WAV 파일로 저장합니다.
    
    Args:
        video_path (str): 입력 비디오 파일 경로.
        audio_path (str): 출력 오디오 파일 경로.
    """
    try:
        video_clip = VideoFileClip(video_path)
        # WAV 형식(pcm_s16le 코덱)은 비압축이므로 호환성이 높습니다.
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        video_clip.close()
        return True
    except Exception as e:
        print(f"오디오 추출 중 오류 발생: {e}")
        return False

# --- 3. 핵심 전사 파이프라인 모듈 ---

def transcribe_video_audio(audio_path):
    """
    WhisperX를 사용하여 오디오 파일을 전사하고 화자를 분리합니다.
    
    Args:
        audio_path (str): 전사할 오디오 파일 경로.
        
    Returns:
        dict: 전사, 정렬, 화자 분리 정보가 포함된 결과 딕셔너리.
    """
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    
    audio = whisperx.load_audio(audio_path)
    
    # 1. 전사 (언어 자동 감지)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    detected_language = result["language"]
    
    # 메모리 정리
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 단어 수준 타임스탬프 정렬
    # 감지된 언어에 맞는 정렬 모델 로드
    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
    
    # 메모리 정리
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 화자 분리
    if HF_TOKEN:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
    return result

# --- 4. 출력 포맷팅 모듈 ---

def format_timestamp(seconds):
    """
    초 단위 시간을 형식의 문자열로 변환합니다.
    
    Args:
        seconds (float): 시간(초).
        
    Returns:
        str: 포맷팅된 타임스탬프 문자열.
    """
    delta = datetime.timedelta(seconds=seconds)
    # timedelta는 마이크로초를 반환하므로 밀리초로 변환하기 위해 1000으로 나눕니다.
    milliseconds = delta.microseconds // 1000
    # str(delta)는 'H:MM:SS' 또는 'D days, H:MM:SS' 형식이므로, 마지막 부분만 사용합니다.
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def format_transcript(result):
    """
    WhisperX 결과 딕셔너리를 지정된 텍스트 형식으로 변환합니다.
    
    Args:
        result (dict): WhisperX의 최종 결과.
        
    Returns:
        str: 최종 포맷팅된 전사 텍스트.
    """
    output_lines = []
    
    # 'segments' 키가 결과에 있는지 확인
    if "segments" not in result or not result["segments"]:
        return "전사된 내용이 없습니다."

    for segment in result["segments"]:
        # 각 세그먼트를 별도의 줄로 처리
        start_time = segment['start']
        speaker = segment.get('speaker', '화자_알수없음')
        text = segment['text'].strip()

        timestamp_str = format_timestamp(start_time)
        formatted_line = f"[{timestamp_str}] {speaker}: {text}"
        output_lines.append(formatted_line)

    return "\n".join(output_lines)

# --- 5. 메인 실행 블록 ---

def main():
    """
    메인 실행 함수. video_input 폴더의 모든 mp4 파일을 처리합니다.
    """
    # 출력 디렉터리 생성
    os.makedirs(TRANSCRIPT_OUTPUT_DIR, exist_ok=True)
    
    # 비디오 파일 목록 가져오기
    video_files = glob.glob(os.path.join(VIDEO_INPUT_DIR, '*.mp4'))
    
    if not video_files:
        print(f"'{VIDEO_INPUT_DIR}' 폴더에 처리할.mp4 파일이 없습니다.")
        return

    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))
        temp_audio_path = os.path.join(VIDEO_INPUT_DIR, f"{base_name}_temp_audio.wav")
        output_txt_path = os.path.join(TRANSCRIPT_OUTPUT_DIR, f"{base_name}_transcript.txt")
        
        # 1. 오디오 추출
        if not extract_audio(video_path, temp_audio_path):
            print(f"'{base_name}.mp4' 처리 실패: 오디오 추출 불가.")
            continue
            
        # 2. 전사 및 화자 분리
        try:
            transcription_result = transcribe_video_audio(temp_audio_path)
            
            # 3. 결과 포맷팅
            formatted_text = format_transcript(transcription_result)
            
            # 4. 텍스트 파일로 저장 (UTF-8 인코딩)
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            print(f"전사 결과가 '{output_txt_path}'에 저장되었습니다.")
            
        except Exception as e:
            print(f"'{base_name}.mp4' 처리 중 오류 발생: {e}")
        finally:
            # 5. 임시 오디오 파일 삭제
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()