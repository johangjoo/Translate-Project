# transcribe_diarize_with_cleanup.py
import os, argparse, pathlib
from pathlib import Path
import torch, whisperx
from whisperx.diarize import DiarizationPipeline
import huggingface_hub
import gc
import psutil
import time

huggingface_hub.login("hf_")

# Windows 전용: PyTorch DLL 경로 우선
try:
    os.add_dll_directory(str(pathlib.Path(torch.__file__).parents[1] / "lib"))
except Exception:
    pass

# 옵션: 속도 향상(TF32). 필요 없으면 주석 처리.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class WhisperXProcessor:
    """WhisperX 처리기 - 메모리 관리 및 초기화 기능 포함"""
    
    def __init__(self, whisper_model="large-v3", device="cpu", compute_type="int8", hf_token=None):
        self.whisper_model = whisper_model
        self.device = device
        self.compute_type = compute_type
        self.hf_token = hf_token
        
        # 모델 인스턴스들
        self.model = None
        self.align_model = None
        self.align_meta = None
        self.diar_pipe = None
        self.current_language = None
        
        # 메모리 모니터링
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
        print(f"🚀 WhisperX 프로세서 초기화")
        print(f"   - 모델: {whisper_model}")
        print(f"   - 디바이스: {device}")
        print(f"   - 정밀도: {compute_type}")
        print(f"   - 초기 메모리: {self.initial_memory:.1f} MB")
    
    def get_memory_usage(self):
        """현재 메모리 사용량 반환 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_gpu_memory_usage(self):
        """GPU 메모리 사용량 반환 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def cleanup_models(self):
        """모든 모델과 캐시 정리"""
        print("🧹 모델 정리 중...")
        
        # 모델 인스턴스 삭제
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
            
        if self.align_meta is not None:
            del self.align_meta
            self.align_meta = None
        
        if self.diar_pipe is not None:
            del self.diar_pipe
            self.diar_pipe = None
        
        self.current_language = None
        
        # 가비지 컬렉션
        gc.collect()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 메모리 사용량 출력
        current_memory = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()
        print(f"   - 현재 메모리: {current_memory:.1f} MB")
        if torch.cuda.is_available():
            print(f"   - GPU 메모리: {gpu_memory:.1f} MB")
    
    def load_whisper_model(self):
        """Whisper 모델 로드"""
        if self.model is None:
            print(f"📥 Whisper 모델 로드: {self.whisper_model}")
            try:
                self.model = whisperx.load_model(
                    self.whisper_model, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
            except Exception as e:
                print(f"⚠️  모델 로드 실패, 기본 설정으로 재시도: {e}")
                self.model = whisperx.load_model(
                    self.whisper_model, 
                    device="cpu", 
                    compute_type="int8"
                )
    
    def load_align_model(self, language_code):
        """정렬 모델 로드 (언어별 캐싱)"""
        if self.current_language != language_code or self.align_model is None:
            print(f"📥 정렬 모델 로드: {language_code}")
            
            # 기존 정렬 모델 정리
            if self.align_model is not None:
                del self.align_model
                del self.align_meta
                gc.collect()
            
            self.align_model, self.align_meta = whisperx.load_align_model(
                language_code=language_code, 
                device=self.device
            )
            self.current_language = language_code
    
    def load_diarization_pipeline(self):
        """화자 분리 파이프라인 로드"""
        if self.diar_pipe is None:
            print("📥 화자 분리 파이프라인 로드")
            try:
                self.diar_pipe = DiarizationPipeline(
                    use_auth_token=self.hf_token, 
                    device=self.device
                )
            except OSError as e:
                print(f"⚠️  GPU 화자분리 실패, CPU로 폴백: {e}")
                self.diar_pipe = DiarizationPipeline(
                    use_auth_token=self.hf_token, 
                    device="cpu"
                )
    
    def process_single_file(self, audio_path, output_dir, batch_size=8, 
                          min_speakers=None, max_speakers=None):
        """단일 파일 처리"""
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        
        print(f"\n🎵 처리 중: {audio_path.name}")
        start_time = time.time()
        
        try:
            # 1. 오디오 로드
            audio = whisperx.load_audio(str(audio_path))
            
            # 2. Whisper 모델 로드 및 전사
            self.load_whisper_model()
            
            try:
                result = self.model.transcribe(audio, batch_size=batch_size)
            except RuntimeError as e:
                # cuBLAS 오류 처리
                if "cublas" in str(e).lower() and "int8" in self.compute_type:
                    print("⚠️  cuBLAS INT8 오류, FP16으로 재시도")
                    self.cleanup_models()
                    self.compute_type = "float16"
                    self.load_whisper_model()
                    result = self.model.transcribe(audio, batch_size=batch_size)
                else:
                    raise
            
            language = result.get("language", "ko")
            print(f"   - 감지된 언어: {language}")
            
            # 3. 정렬 모델 로드 및 정렬
            self.load_align_model(language)
            aligned_result = whisperx.align(
                result["segments"], 
                self.align_model, 
                self.align_meta, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            # 4. 화자 분리
            self.load_diarization_pipeline()
            
            diar_kwargs = {}
            if min_speakers is not None:
                diar_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diar_kwargs["max_speakers"] = max_speakers
            
            diarization_result = self.diar_pipe(audio, **diar_kwargs)
            final_result = whisperx.assign_word_speakers(diarization_result, aligned_result)
            
            # 5. 결과 저장
            output_path = self.save_transcript(final_result, audio_path, output_dir, language)
            
            # 6. 처리 완료 정보
            elapsed_time = time.time() - start_time
            current_memory = self.get_memory_usage()
            print(f"✅ 완료: {audio_path.name}")
            print(f"   - 출력: {output_path.name}")
            print(f"   - 처리 시간: {elapsed_time:.1f}초")
            print(f"   - 메모리 사용량: {current_memory:.1f} MB")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 오류: {audio_path.name} - {e}")
            return None
        
        finally:
            # 오디오 메모리 정리
            if 'audio' in locals():
                del audio
            gc.collect()
    
    def save_transcript(self, result, audio_path, output_dir, language):
        """전사 결과를 파일로 저장"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 화자 매핑 및 라인 생성
        speaker_map = {}
        lines = []
        
        for segment in sorted(result["segments"], key=lambda x: x.get("start", 0.0)):
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            
            speaker = segment.get("speaker", "UNK")
            if speaker not in speaker_map:
                speaker_map[speaker] = f"화자{len(speaker_map) + 1}"
            
            start_time = segment.get("start", 0.0)
            time_str = self.format_time(start_time)
            lines.append(f"[{time_str}] {speaker_map[speaker]} : {text}")
        
        # 파일 저장
        output_path = output_dir / f"{audio_path.stem}_transcript_{language}.txt"
        with output_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        
        return output_path
    
    def format_time(self, seconds):
        """시간을 MM:SS 형식으로 변환"""
        if not isinstance(seconds, (int, float)) or seconds != seconds or seconds < 0:
            seconds = 0.0
        
        minutes = int(seconds // 60)
        secs = int(round(seconds - minutes * 60))
        
        if secs == 60:
            minutes += 1
            secs = 0
        
        return f"{minutes:02d}:{secs:02d}"
    
    def process_batch(self, input_path, output_dir, batch_size=8, 
                     min_speakers=None, max_speakers=None, 
                     file_pattern="*.mp4", cleanup_interval=5):
        """배치 처리 - 여러 파일을 순차적으로 처리"""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        # 파일 목록 수집
        if input_path.is_file():
            files = [input_path]
        else:
            # 여러 오디오 형식 지원
            audio_extensions = ["*.mp4", "*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac", "*.ogg"]
            files = []
            
            if file_pattern != "*.mp4":
                files = sorted(input_path.glob(file_pattern))
            else:
                for ext in audio_extensions:
                    files.extend(input_path.glob(ext))
                files = sorted(set(files))
        
        if not files:
            print(f"❌ {input_path}에서 오디오 파일을 찾을 수 없습니다.")
            return
        
        print(f"\n📁 배치 처리 시작: {len(files)}개 파일")
        for i, file_path in enumerate(files, 1):
            print(f"  {i}. {file_path.name}")
        
        # 파일별 처리
        successful = 0
        failed = 0
        
        for i, file_path in enumerate(files, 1):
            print(f"\n{'='*60}")
            print(f"📊 진행률: {i}/{len(files)} ({i/len(files)*100:.1f}%)")
            
            result = self.process_single_file(
                file_path, output_dir, batch_size, min_speakers, max_speakers
            )
            
            if result:
                successful += 1
            else:
                failed += 1
            
            # 주기적으로 모델 정리 (메모리 누수 방지)
            if i % cleanup_interval == 0 or i == len(files):
                print(f"\n🔄 주기적 정리 ({i}/{len(files)})")
                self.cleanup_models()
                
                # 잠시 대기 (시스템 안정화)
                time.sleep(2)
        
        # 최종 결과
        print(f"\n{'='*60}")
        print(f"🎉 배치 처리 완료!")
        print(f"   - 성공: {successful}개")
        print(f"   - 실패: {failed}개")
        print(f"   - 총 처리: {len(files)}개")
        
        # 최종 정리
        self.cleanup_models()

def main():
    ap = argparse.ArgumentParser(
        description="WhisperX 배치 처리 (메모리 관리 및 초기화 기능 포함)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default="video_input", help="오디오/비디오 파일 또는 폴더")
    ap.add_argument("--out_dir", default="input", help="출력 폴더")
    ap.add_argument("--whisper_model", default="large-v3", help="Whisper 모델명")
    ap.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    ap.add_argument("--min_speakers", type=int, default=None, help="최소 화자 수")
    ap.add_argument("--max_speakers", type=int, default=None, help="최대 화자 수")
    ap.add_argument("--hf_token", default=None, help="Hugging Face 토큰")
    ap.add_argument("--force_cpu", action="store_true", help="CPU 강제 사용")
    ap.add_argument("--compute_type", default=None, help="연산 정밀도")
    ap.add_argument("--ext", default="*.mp4", help="파일 확장자 패턴")
    ap.add_argument("--cleanup_interval", type=int, default=5, help="정리 주기 (파일 개수)")
    
    args = ap.parse_args()
    
    # 토큰 설정
    token = (
        args.hf_token
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    
    # 디바이스 설정
    device = "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 정밀도 설정
    if args.compute_type:
        compute_type = args.compute_type
    else:
        compute_type = "float16" if device == "cuda" else "int8"
    
    # 프로세서 생성 및 실행
    processor = WhisperXProcessor(
        whisper_model=args.whisper_model,
        device=device,
        compute_type=compute_type,
        hf_token=token
    )
    
    try:
        processor.process_batch(
            input_path=args.src,
            output_dir=args.out_dir,
            batch_size=args.batch_size,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            file_pattern=args.ext,
            cleanup_interval=args.cleanup_interval
        )
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
    finally:
        print("\n🧹 최종 정리...")
        processor.cleanup_models()

if __name__ == "__main__":
    main()
