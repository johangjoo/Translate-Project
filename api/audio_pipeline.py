#!/usr/bin/env python3
"""
음성 파일 노이즈 제거 → Whisper STT → 음성 특성 기반 화자분리 → 텍스트 저장 파이프라인

필요한 라이브러리:
- librosa: 고급 음성 특성 추출 (pip install librosa)
- scikit-learn: 클러스터링 (pip install scikit-learn)
- numpy: 기본 수치 연산 (pip install numpy)
- soundfile: 임시 WAV 파일 저장 (pip install soundfile)
- tempfile: 임시 파일 생성 (pip install tempfile)
"""

import os
import sys
import glob
import logging
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import torch
import torchaudio
import whisper
import soundfile as sf
import numpy as np
from typing import Optional, Dict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SpeechBrain import를 조건부로 처리
try:
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SPEECHBRAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SpeechBrain 로딩 실패: {e}")
    SPEECHBRAIN_AVAILABLE = False

# WhisperX + pyannote diarization 사용 여부
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WhisperX 로딩 실패 (pyannote 화자분리는 비활성화됩니다): {e}")
    WHISPERX_AVAILABLE = False

class AudioPipeline:
    """음성 및 비디오 파이프라인 클래스"""
    
    def __init__(self, use_gpu=True, target_language=None):
        """
        파이프라인 초기화
        
        Args:
            use_gpu (bool): GPU 사용 여부
            target_language (str): 대상 언어 ('ko', 'ja', 'en', None=자동감지)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.target_language = target_language
        
        # WhisperX + pyannote 설정 (환경변수에서 토큰 읽기)
        self.pyannote_auth_token = os.getenv("PYANNOTE_AUTH_TOKEN", None)
        self.use_whisperx_diarization = True
        
        # 최대 화자 수 설정 (1~10 범위에서 사용)
        self.max_speakers = 5

        # VAD(Voice Activity Detection) 사용 여부
        # True이면 노이즈 제거 이후에 앞/뒤 무음을 잘라서 STT 효율을 높입니다.
        self.enable_vad = True

        # 폴더 경로 설정
        self.audio_input_dir = Path("audio_input")
        self.audio_out_dir = Path("audio_out") 
        self.audio_output_dir = Path("audio_output")
        self.script_output_dir = Path("script_output")
        
        # 폴더 생성
        self._create_directories()
        
        # 모델 초기화
        self.denoiser = None
        self.whisper_model = None
        self.speaker_encoder = None  # ECAPA-VOXCELEB 화자분리 모델
        
        # 지원 언어 정보 (한국어, 일본어, 영어만 지원)
        self.supported_languages = {
            'ko': '한국어',
            'ja': '日本語',
            'en': 'English'
        }
        
        # 지원 오디오 파일 형식 (영상은 Electron 쪽에서 WAV로 변환됨)
        self.audio_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
        
        logger.info(f"파이프라인 초기화 완료 - 디바이스: {self.device}")
        if target_language:
            lang_name = self.supported_languages.get(target_language, target_language)
            logger.info(f"대상 언어: {lang_name} ({target_language})")
        else:
            logger.info("언어: 자동 감지 모드")
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.audio_input_dir, self.audio_out_dir, 
                         self.audio_output_dir, self.script_output_dir]:
            directory.mkdir(exist_ok=True)
            logger.info(f"디렉토리 생성/확인: {directory}")
    
    def _is_audio_file(self, file_path):
        """오디오 파일인지 확인"""
        return Path(file_path).suffix.lower() in self.audio_formats
    
    def _load_denoiser(self):
        """노이즈 제거 모델 로드"""
        if self.denoiser is None:
            if not SPEECHBRAIN_AVAILABLE:
                logger.warning("SpeechBrain을 사용할 수 없어 대안 방법 사용")
                self._load_denoiser_alternative()
                return
                
            try:
                logger.info("노이즈 제거 모델 로딩 중...")
                
                # Windows 권한 문제 해결을 위해 LocalStrategy 사용
                import os
                os.environ['SPEECHBRAIN_CACHE_STRATEGY'] = 'LOCAL'
                
                # 절대 경로로 savedir 설정
                savedir = os.path.abspath("pretrained_models/metricgan-plus-voicebank")
                
                self.denoiser = SpectralMaskEnhancement.from_hparams(
                    source="speechbrain/metricgan-plus-voicebank",
                    savedir=savedir,
                    run_opts={"device": self.device}
                )
                logger.info("노이즈 제거 모델 로딩 완료")
            except Exception as e:
                logger.error(f"노이즈 제거 모델 로딩 실패: {e}")
                logger.info("대안 방법으로 재시도...")
                try:
                    self._load_denoiser_alternative()
                except Exception as e2:
                    logger.error(f"대안 방법도 실패: {e2}")
                    raise e
    
    def _load_denoiser_alternative(self):
        """대안 노이즈 제거 방법 (간단한 스펙트럼 필터링)"""
        logger.info("대안 노이즈 제거 방법 사용 (간단한 필터링)")
        self.denoiser = "simple_filter"  # 플래그로 사용
    
    def _simple_denoise(self, waveform, sample_rate):
        """간단한 노이즈 제거 (스펙트럼 필터링)"""
        try:
            # 간단한 고역 통과 필터 적용
            from scipy.signal import butter, filtfilt
            
            # 80Hz 이하 저주파 노이즈 제거
            nyquist = sample_rate / 2
            low_cutoff = 80 / nyquist
            b, a = butter(4, low_cutoff, btype='high')
            
            # 필터 적용
            filtered = filtfilt(b, a, waveform.numpy())
            
            # 정규화
            filtered = filtered / np.max(np.abs(filtered)) * 0.8
            
            return torch.from_numpy(filtered).float()
            
        except ImportError:
            logger.warning("scipy가 설치되지 않아 간단한 정규화만 적용")
            # 간단한 정규화만 적용
            normalized = waveform / torch.max(torch.abs(waveform)) * 0.8
            return normalized

    def _apply_vad(self, waveform, sample_rate, energy_threshold: float = 0.02):
        """
        매우 단순한 에너지 기반 VAD.
        - 입력: 단일 채널 waveform (1, N)
        - 출력: 앞/뒤 무음이 잘려진 waveform (내부 긴 무음은 유지)

        Args:
            waveform: 1채널 오디오 텐서 (1, N)
            sample_rate: 샘플링 레이트 (Hz)
            energy_threshold: 최대 에너지 대비 말소리로 볼 최소 비율 (0.0 ~ 1.0)
        """
        try:
            # 모노 보장
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 아주 짧은 경우는 그대로 사용
            if waveform.size(-1) < sample_rate * 0.3:
                return waveform

            window_size = int(sample_rate * 0.03)  # 30 ms
            hop_size = max(1, window_size // 2)
            num_samples = waveform.size(-1)

            energies = []
            for start in range(0, num_samples, hop_size):
                end = min(start + window_size, num_samples)
                frame = waveform[..., start:end]
                if frame.numel() == 0:
                    break
                # RMS 에너지
                energy = torch.sqrt(torch.mean(frame ** 2))
                energies.append(energy.item())

            if not energies:
                return waveform

            energies_tensor = torch.tensor(energies)
            max_energy = float(energies_tensor.max())
            if max_energy <= 0:
                return waveform

            threshold = max_energy * float(energy_threshold)
            speech_indices = (energies_tensor > threshold).nonzero(as_tuple=False).flatten()
            if speech_indices.numel() == 0:
                # 전부 무음으로 판단되면 원본 유지
                return waveform

            first_idx = int(speech_indices[0])
            last_idx = int(speech_indices[-1])
            start_sample = max(0, first_idx * hop_size)
            end_sample = min(num_samples, (last_idx + 1) * hop_size)

            trimmed = waveform[..., start_sample:end_sample]
            logger.info(
                f"VAD 트리밍: {num_samples / sample_rate:.2f}s → "
                f"{trimmed.size(-1) / sample_rate:.2f}s"
            )
            return trimmed
        except Exception as e:
            logger.warning(f"VAD 처리 중 오류, 원본 waveform 사용: {e}")
            return waveform

    def _load_whisper(self, model_size="large-v3"):
        """Whisper 모델 로드"""
        if self.whisper_model is None:
            try:
                logger.info(f"Whisper 모델 ({model_size}) 로딩 중...")
                self.whisper_model = whisper.load_model(model_size, device=self.device)
                logger.info("Whisper 모델 로딩 완료")
            except Exception as e:
                logger.error(f"Whisper 모델 로딩 실패: {e}")
                raise
    
    def _load_speaker_encoder(self):
        """ECAPA-VOXCELEB 화자분리 모델 로드"""
        if self.speaker_encoder is None:
            if not SPEECHBRAIN_AVAILABLE:
                logger.warning("SpeechBrain을 사용할 수 없어 화자분리 기능을 사용할 수 없습니다")
                return False
                
            try:
                logger.info("ECAPA-VOXCELEB 화자분리 모델 로딩 중...")
                
                # 권한 문제 우회를 위해 임시 디렉토리 사용
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="speechbrain_")
                
                # ECAPA-VOXCELEB 모델 로드
                self.speaker_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=temp_dir,
                    run_opts={"device": self.device}
                )
                logger.info("ECAPA-VOXCELEB 화자분리 모델 로딩 완료")
                return True
            except Exception as e:
                logger.error(f"화자분리 모델 로딩 실패: {e}")
                logger.info("권한 문제일 수 있습니다. 관리자 권한으로 실행하거나 규칙 기반 방식을 사용합니다.")
                self.speaker_encoder = None
                return False
        return True
    
    def denoise_audio(self, input_file, output_file):
        """
        오디오 파일 노이즈 제거
        
        Args:
            input_file (str): 입력 파일 경로
            output_file (str): 출력 파일 경로
        """
        try:
            logger.info(f"노이즈 제거 시작: {input_file}")
            
            # 모델 로드
            self._load_denoiser()
            
            # 오디오 파일 로드
            waveform, sample_rate = torchaudio.load(input_file)
            
            # 모노 채널로 변환 (필요한 경우)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 샘플링 레이트를 16kHz로 변환 (SpeechBrain 모델 요구사항)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # 노이즈 제거 수행
            if self.denoiser == "simple_filter":
                # 대안 방법 사용
                logger.info("간단한 필터링 방법으로 노이즈 제거")
                enhanced_waveform = self._simple_denoise(waveform.squeeze(0), sample_rate)
                enhanced_waveform = enhanced_waveform.unsqueeze(0)
            else:
                # SpeechBrain 모델 사용
                waveform = waveform.to(self.device)
                enhanced_waveform = self.denoiser.enhance_batch(waveform.unsqueeze(0))
                enhanced_waveform = enhanced_waveform.squeeze(0).cpu()

            # VAD 기반 앞/뒤 무음 제거 (옵션)
            if getattr(self, "enable_vad", False):
                try:
                    logger.info("VAD 기반 무음 구간 트리밍 수행")
                    enhanced_waveform = self._apply_vad(enhanced_waveform, sample_rate)
                except Exception as e:
                    logger.warning(f"VAD 적용 실패, 원본 waveform 사용: {e}")

            # 출력 파일 저장
            torchaudio.save(output_file, enhanced_waveform, sample_rate)
            
            logger.info(f"노이즈 제거 완료: {output_file}")
            
        except Exception as e:
            logger.error(f"노이즈 제거 실패 ({input_file}): {e}")
            raise
    
    def transcribe_audio(self, audio_file, output_text_file, srt_file=None, diarization_audio_file=None, enable_timestamps=True):
        """
        오디오 파일을 텍스트로 변환 (STT)
        
        Args:
            audio_file (str): 오디오 파일 경로
            output_text_file (str): 출력 텍스트 파일 경로
            srt_file (str, optional): SRT 자막 파일 경로
            diarization_audio_file (str, optional): 화자분리용 오디오 파일 경로
        """
        try:
            logger.info(f"STT 처리 시작: {audio_file}")
            
            # Whisper 모델 로드
            self._load_whisper()
            
            # 음성 인식 옵션 (단어별 타임스탬프 포함)
            # condition_on_previous_text=False 로 설정하여
            # 긴 침묵 이후 동일 문장 반복/환각을 줄인다.
            transcribe_options = {
                "word_timestamps": False,
                "verbose": True,
                "condition_on_previous_text": False,
            }
            
            # 언어 설정 (유효한 언어 코드만 사용)
            if self.target_language and self.target_language in self.supported_languages:
                transcribe_options["language"] = self.target_language
                lang_name = self.supported_languages.get(self.target_language, self.target_language)
                logger.info(f"지정된 언어로 STT 처리: {lang_name}")
            else:
                if self.target_language:
                    logger.warning(f"⚠️ 유효하지 않은 언어 코드 무시: {self.target_language} (자동 감지 모드로 전환)")
                logger.info("언어 자동 감지로 STT 처리")
            
            # 긴 오디오는 내부적으로 chunk 단위로 나누어 처리
            total_duration = None
            sample_rate = None
            num_frames = None
            try:
                info = torchaudio.info(audio_file)
                sample_rate = info.sample_rate
                num_frames = info.num_frames
                if sample_rate and num_frames:
                    total_duration = num_frames / float(sample_rate)
            except Exception as e:
                logger.warning(f"오디오 메타데이터 확인 실패, 단일 파일로 처리합니다: {e}")
            
            # chunk 기준 (초 단위, 10분)
            chunk_duration = 10800.0
            
            if total_duration is None or total_duration <= chunk_duration:
                # 기존 방식: 전체 파일을 한 번에 처리
                result = self.whisper_model.transcribe(str(audio_file), **transcribe_options)
            else:
                logger.info(
                    f"긴 오디오 감지 ({total_duration/60.0:.1f}분) - "
                    f"{chunk_duration/60.0:.0f}분 단위 chunk로 분할 처리"
                )

                merged_segments = []
                text_parts = []
                chunk_results = []

                # 샘플 기준 chunk 크기
                chunk_samples = int(chunk_duration * sample_rate)
                offset_sec = 0.0

                start_frame = 0
                while start_frame < num_frames:
                    remaining = num_frames - start_frame
                    this_frames = min(chunk_samples, remaining)

                    logger.info(
                        f"chunk STT 처리: start_frame={start_frame}, "
                        f"frames={this_frames}, offset={offset_sec:.2f}s"
                    )

                    # 해당 chunk만 로드
                    waveform, sr = torchaudio.load(audio_file, frame_offset=start_frame, num_frames=this_frames)

                    # 모노 변환
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    # numpy로 변환하여 임시 wav 파일로 저장 후 Whisper 호출
                    audio_np = waveform.squeeze(0).numpy()

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        tmp_path = tmp_wav.name
                    try:
                        sf.write(tmp_path, audio_np, sr)
                        chunk_result = self.whisper_model.transcribe(tmp_path, **transcribe_options)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                    chunk_results.append(chunk_result)

                    # 텍스트 누적
                    chunk_text = chunk_result.get("text", "").strip()
                    if chunk_text:
                        text_parts.append(chunk_text)

                    # 세그먼트 타임스탬프에 offset 적용 후 병합
                    if "segments" in chunk_result:
                        for seg in chunk_result["segments"]:
                            seg_copy = dict(seg)
                            seg_copy["start"] = float(seg_copy.get("start", 0.0)) + offset_sec
                            seg_copy["end"] = float(seg_copy.get("end", seg_copy.get("start", 0.0))) + offset_sec

                            if "words" in seg_copy:
                                words = []
                                for w in seg_copy["words"]:
                                    w_copy = dict(w)
                                    w_copy["start"] = float(w_copy.get("start", 0.0)) + offset_sec
                                    w_copy["end"] = float(w_copy.get("end", w_copy.get("start", 0.0))) + offset_sec
                                    words.append(w_copy)
                                seg_copy["words"] = words

                            merged_segments.append(seg_copy)

                    # 다음 chunk로 이동
                    start_frame += this_frames
                    offset_sec += this_frames / float(sample_rate)

                if not chunk_results:
                    raise RuntimeError("chunk STT 결과가 비어 있습니다.")

                merged_text = " ".join(text_parts).strip()
                merged_language = chunk_results[0].get("language", "unknown")

                result = {
                    "text": merged_text,
                    "segments": merged_segments,
                    "language": merged_language,
                    "duration": total_duration or (offset_sec if offset_sec > 0 else None),
                }

            # 결과 텍스트 추출
            transcribed_text = result["text"].strip()
            
            # 타임스탬프가 포함된 텍스트 파일로 저장
            self._save_transcript_with_timestamps(audio_file, output_text_file, result)
            
            # 간단한 타임스탬프+화자 정보 파일 생성 (옵션에 따라)
            simple_file = Path(output_text_file).parent / f"{Path(output_text_file).stem}_simple.txt"
            if enable_timestamps and diarization_audio_file:
                # 타임스탬프와 화자분리 모두 활성화
                self._save_simple_transcript(simple_file, result, diarization_audio_file)
                logger.info(f"간단한 전사 파일 생성 (타임스탬프+화자분리): {simple_file}")
            elif enable_timestamps:
                # 타임스탬프만 활성화
                self._save_simple_transcript_timestamps_only(simple_file, result)
                logger.info(f"간단한 전사 파일 생성 (타임스탬프만): {simple_file}")
            elif diarization_audio_file:
                # 화자분리만 활성화
                self._save_simple_transcript_speakers_only(simple_file, result, diarization_audio_file)
                logger.info(f"간단한 전사 파일 생성 (화자분리만): {simple_file}")
            else:
                # 둘 다 비활성화 - 순수 텍스트만
                self._save_simple_transcript_text_only(simple_file, result)
                logger.info(f"간단한 전사 파일 생성 (텍스트만): {simple_file}")
            
            # SRT 자막 파일 생성 (요청된 경우)
            if srt_file:
                diarization_source = diarization_audio_file or audio_file
                self._save_srt_file(srt_file, result, diarization_source)
                logger.info(f"SRT 자막 파일 생성: {srt_file}")
            
            logger.info(f"STT 처리 완료: {output_text_file}")
            logger.info(f"인식된 텍스트: {transcribed_text[:100]}...")
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"STT 처리 실패 ({audio_file}): {e}")
            raise
    
    def _save_transcript_with_timestamps(self, audio_file, output_text_file, result):
        """타임스탬프가 포함된 전사 결과 저장"""
        with open(output_text_file, 'w', encoding='utf-8') as f:
            # 헤더 정보
            f.write(f"파일명: {Path(audio_file).name}\n")
            f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"언어: {result.get('language', 'unknown')}\n")
            f.write(f"총 길이: {self._format_time(result.get('duration', 0))}\n")
            f.write("=" * 60 + "\n\n")
            
            # 전체 텍스트
            f.write("📝 전체 텍스트\n")
            f.write("-" * 30 + "\n")
            f.write(result["text"].strip() + "\n\n")
            
            # 세그먼트별 타임스탬프
            f.write("⏰ 타임스탬프별 텍스트\n")
            f.write("-" * 30 + "\n")
            
            if "segments" in result:
                for i, segment in enumerate(result["segments"], 1):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"[{start_time} → {end_time}] {text}\n")
                
                # 단어별 타임스탬프 (가능한 경우)
                f.write("\n🔤 단어별 타임스탬프\n")
                f.write("-" * 30 + "\n")
                
                for segment in result["segments"]:
                    if "words" in segment:
                        for word_info in segment["words"]:
                            start_time = self._format_time(word_info["start"])
                            end_time = self._format_time(word_info["end"])
                            word = word_info["word"].strip()
                            confidence = word_info.get("probability", 0)
                            
                            f.write(f"[{start_time}-{end_time}] {word} (신뢰도: {confidence:.2f})\n")
            else:
                f.write("타임스탬프 정보를 사용할 수 없습니다.\n")
    
    def _format_time(self, seconds):
        """초를 MM:SS.mmm 형식으로 변환"""
        if seconds is None:
            return "00:00.000"
        
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    
    def _save_srt_file(self, srt_file, result, audio_file=None):
        """SRT 자막 파일 생성 (화자 정보 포함)"""
        with open(srt_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                # 음성 특성 기반 화자분리 사용
                if audio_file:
                    # 현재 오디오 파일 정보를 임시 저장
                    self._current_audio_file = audio_file
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                    # 임시 정보 제거
                    delattr(self, '_current_audio_file')
                else:
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                
                subtitle_index = 1
                for i, segment in enumerate(result["segments"]):
                    start_time = self._format_srt_time(segment["start"])
                    end_time = self._format_srt_time(segment["end"])
                    text = segment["text"].strip()
                    
                    if not text:
                        continue
                    
                    # 할당된 화자 사용
                    speaker_name = speaker_assignments[i]
                    
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{speaker_name}: {text}\n\n")
                    subtitle_index += 1
            else:
                # 세그먼트 정보가 없는 경우 전체 텍스트를 하나의 자막으로
                duration = result.get("duration", 60)  # 기본 60초
                start_time = self._format_srt_time(0)
                end_time = self._format_srt_time(duration)
                text = result["text"].strip()
                
                f.write("1\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"화자A: {text}\n\n")
    
    def _format_srt_time(self, seconds):
        """초를 SRT 형식 (HH:MM:SS,mmm)으로 변환"""
        if seconds is None:
            return "00:00:00,000"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs % 1) * 1000)
        secs = int(secs)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _save_simple_transcript(self, simple_file, result, audio_file=None):
        """간단한 타임스탬프+화자 정보 텍스트 파일 생성"""
        with open(simple_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                # 음성 특성 기반 화자분리 사용
                if audio_file:
                    logger.info("음성 특성 기반 화자분리 시작...")
                    # 현재 오디오 파일 정보를 임시 저장
                    self._current_audio_file = audio_file
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                    # 임시 정보 제거
                    delattr(self, '_current_audio_file')
                else:
                    logger.info("대안 화자 할당 사용...")
                    speaker_assignments = self._assign_smart_speakers(result["segments"])
                
                for i, segment in enumerate(result["segments"]):
                    start_time = self._format_time(segment["start"])
                    text = segment["text"].strip()
                    
                    if not text:
                        continue
                    
                    # 할당된 화자 사용
                    speaker_name = speaker_assignments[i]
                    
                    # [시간] 화자: 텍스트 형식으로 저장
                    f.write(f"[{start_time}] {speaker_name}: {text}\n")
            else:
                # 세그먼트 정보가 없는 경우
                f.write(f"[00:00.000] 화자A: {result['text'].strip()}\n")
    
    def _save_simple_transcript_timestamps_only(self, simple_file, result):
        """타임스탬프만 포함된 간단한 전사 파일 생성"""
        with open(simple_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                for segment in result["segments"]:
                    start_time = self._format_time(segment["start"])
                    text = segment["text"].strip()
                    
                    if not text:
                        continue
                    
                    # [시간] 텍스트 형식으로 저장 (화자 정보 없음)
                    f.write(f"[{start_time}] {text}\n")
            else:
                # 세그먼트 정보가 없는 경우
                f.write(f"[00:00.000] {result['text'].strip()}\n")
    
    def _save_simple_transcript_speakers_only(self, simple_file, result, audio_file):
        """화자분리만 포함된 간단한 전사 파일 생성"""
        with open(simple_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                # 음성 특성 기반 화자분리 사용
                logger.info("음성 특성 기반 화자분리 시작...")
                # 현재 오디오 파일 정보를 임시 저장
                self._current_audio_file = audio_file
                speaker_assignments = self._assign_smart_speakers(result["segments"])
                # 임시 정보 제거
                delattr(self, '_current_audio_file')
                
                for i, segment in enumerate(result["segments"]):
                    text = segment["text"].strip()
                    
                    if not text:
                        continue
                    
                    # 할당된 화자 사용
                    speaker_name = speaker_assignments[i]
                    
                    # 화자: 텍스트 형식으로 저장 (타임스탬프 없음)
                    f.write(f"{speaker_name}: {text}\n")
            else:
                # 세그먼트 정보가 없는 경우
                f.write(f"화자A: {result['text'].strip()}\n")
    
    def _save_simple_transcript_text_only(self, simple_file, result):
        """순수 텍스트만 포함된 간단한 전사 파일 생성"""
        with open(simple_file, 'w', encoding='utf-8') as f:
            if "segments" in result:
                for segment in result["segments"]:
                    text = segment["text"].strip()
                    
                    if not text:
                        continue
                    
                    # 순수 텍스트만 저장 (타임스탬프, 화자 정보 없음)
                    f.write(f"{text}\n")
            else:
                # 세그먼트 정보가 없는 경우
                f.write(f"{result['text'].strip()}\n")
    
    def _assign_smart_speakers(self, segments):
        """화자 분리: WhisperX + pyannote를 우선 사용, 실패 시 기존 음성 특성/규칙 기반 사용"""
        if not segments:
            return []
        
        logger.info(f"화자 분리 시작 - 총 {len(segments)}개 세그먼트")
        
        # 세그먼트가 너무 적으면 단일 화자
        if len(segments) <= 1:
            logger.info("세그먼트 1개 이하 - 단일 화자로 처리")
            return ["화자A" for _ in segments]

        # 1순위: WhisperX + pyannote diarization 사용
        audio_file = getattr(self, '_current_audio_file', None)
        if (
            self.use_whisperx_diarization
            and WHISPERX_AVAILABLE
            and self.pyannote_auth_token
            and audio_file
        ):
            try:
                logger.info("WhisperX + pyannote 기반 화자 분리 시도")
                return self._assign_speakers_with_whisperx(audio_file, segments)
            except Exception as e:
                logger.error(f"WhisperX 화자 분리 실패, 기존 방식으로 대체: {e}")

        # 2순위: 기존 음성 특성 기반 방식
        if audio_file:
            return self._voice_feature_based_assignment(audio_file, segments)
        else:
            # 오디오 파일이 없으면 단순 규칙 기반 로직
            return self._fallback_speaker_assignment(segments)

    def _assign_speakers_with_whisperx(self, audio_file, segments):
        """WhisperX + pyannote diarization 결과를 Whisper 세그먼트에 매핑"""
        if not WHISPERX_AVAILABLE:
            raise RuntimeError("WhisperX가 설치되어 있지 않습니다.")
        if not self.pyannote_auth_token:
            raise RuntimeError("PYANNOTE_AUTH_TOKEN이 설정되어 있지 않습니다.")
        
        logger.info("WhisperX diarization 파이프라인 초기화")
        device = self.device
        
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.pyannote_auth_token,
            device=device,
        )
        
        logger.info("WhisperX diarization 실행")
        diarize_result = diarize_model(audio_file)
        
        # pyannote Annotation 객체에서 (start, end, speaker_label) 리스트 추출
        diarization_segments = []
        for turn, _, speaker in diarize_result.itertracks(yield_label=True):
            diarization_segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )
        
        if not diarization_segments:
            logger.warning("WhisperX diarization 결과가 비어 있습니다. 기존 방식 사용")
            return self._voice_feature_based_assignment(audio_file, segments)
        
        # diarization speaker 라벨을 화자A/B/... 로 매핑
        speaker_label_map = {}
        speaker_order = []
        
        def _map_label(raw_label):
            if raw_label not in speaker_label_map:
                speaker_order.append(raw_label)
                idx = len(speaker_order) - 1
                speaker_label_map[raw_label] = f"화자{chr(ord('A') + idx)}"
            return speaker_label_map[raw_label]
        
        # 세그먼트 중심 시간을 기준으로 diarization 세그먼트와 매칭
        speaker_assignments = []
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + 0.1))
            mid = (start + end) / 2.0
            
            matched_speaker = None
            for dseg in diarization_segments:
                if dseg["start"] <= mid <= dseg["end"]:
                    matched_speaker = _map_label(dseg["speaker"])
                    break
            
            # 매칭 실패 시 가장 인접한 diarization 세그먼트 사용
            if matched_speaker is None:
                best_dist = None
                best_label = None
                for dseg in diarization_segments:
                    if mid < dseg["start"]:
                        dist = dseg["start"] - mid
                    elif mid > dseg["end"]:
                        dist = mid - dseg["end"]
                    else:
                        dist = 0.0
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_label = dseg["speaker"]
                if best_label is not None:
                    matched_speaker = _map_label(best_label)

            # 그래도 없으면 기본 화자A
            if matched_speaker is None:
                matched_speaker = "화자A"

            speaker_assignments.append(matched_speaker)

        from collections import Counter
        logger.info(f"WhisperX 기반 화자 분포: {dict(Counter(speaker_assignments))}")
        speaker_assignments = self._smooth_whisperx_speakers(segments, speaker_assignments)

        return speaker_assignments
    
    def _smooth_whisperx_speakers(self, segments, speaker_assignments):
        try:
            from collections import Counter

            if not segments or not speaker_assignments:
                return speaker_assignments

            n = len(segments)
            assignments = list(speaker_assignments)

            for i, (seg, spk) in enumerate(zip(segments, assignments)):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                duration = max(0.0, end - start)

                prev_spk = assignments[i - 1] if i > 0 else None
                next_spk = assignments[i + 1] if i < n - 1 else None

                if (
                    duration < 0.7
                    and prev_spk is not None
                    and next_spk is not None
                    and prev_spk == next_spk
                    and spk != prev_spk
                ):
                    assignments[i] = prev_spk

            counts = Counter(assignments)
            total = sum(counts.values())

            if total == 0:
                return assignments

            main_speakers = {s for s, c in counts.items() if c / total >= 0.15}

            if not main_speakers:
                return assignments

            for i, spk in enumerate(assignments):
                ratio = counts[spk] / total
                if ratio >= 0.05:
                    continue

                neighbors = []
                if i > 0:
                    neighbors.append(assignments[i - 1])
                if i < n - 1:
                    neighbors.append(assignments[i + 1])

                candidates = [s for s in neighbors if s in main_speakers]
                if candidates:
                    assignments[i] = candidates[0]

            return assignments

        except Exception:
            return speaker_assignments
    
    def _voice_feature_based_assignment(self, audio_file, segments):
        """음성 특성(주파수, 피치, 스펙트럼) 기반 화자 분리 + 독백 처리"""
        try:
            logger.info("음성 특성 추출 및 독백 분석 중...")
            
            # 독백 여부 사전 판단
            is_monologue = self._detect_monologue_pattern(segments)
            if is_monologue:
                logger.info("독백 패턴 감지 - 독백 전용 처리 모드")
                return self._handle_monologue_segments(segments)
            
            # 오디오 파일 로드
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # 모노 채널로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 세그먼트별 음성 특성 추출
            voice_features = []
            valid_segments = []
            
            for i, segment in enumerate(segments):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", start_time + 1)
                text = segment.get("text", "").strip()
                
                if not text or end_time <= start_time:
                    continue
                
                # 세그먼트 오디오 추출
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                if start_sample >= waveform.shape[1] or end_sample <= start_sample:
                    continue
                
                segment_audio = waveform[:, start_sample:end_sample]
                
                # 너무 짧은 세그먼트는 건너뛰기 (최소 0.3초)
                if segment_audio.shape[1] < sample_rate * 0.3:
                    continue
                
                # 음성 특성 추출
                features = self._extract_voice_features(segment_audio.squeeze(), sample_rate)
                if features is not None:
                    voice_features.append(features)
                    valid_segments.append((i, segment))
            
            if len(voice_features) < 2:
                logger.info("유효한 음성 특성이 부족 - 단일 화자로 처리")
                return ["화자A" for _ in segments]
            
            # 음성 특성 기반 클러스터링
            speaker_labels = self._cluster_voice_features(voice_features)
            
            # 전체 세그먼트에 화자 할당
            return self._assign_speakers_from_clusters(segments, valid_segments, speaker_labels)
            
        except Exception as e:
            logger.error(f"음성 특성 기반 화자분리 실패: {e}")
            return self._fallback_speaker_assignment(segments)
    
    def _detect_monologue_pattern(self, segments):
        """독백 패턴 감지 (개선된 버전)"""
        try:
            logger.info(f"독백 패턴 분석 중... (세그먼트 수: {len(segments)})")
            
            # 세그먼트가 1개인 경우에만 확실한 독백으로 처리
            if len(segments) <= 1:
                logger.info("세그먼트 1개 이하 - 독백으로 판단")
                return True
            
            # 1. 침묵 시간 분석
            silence_durations = []
            long_silences = 0  # 2초 이상 침묵 (기준 완화)
            very_long_silences = 0  # 5초 이상 침묵
            
            for i in range(1, len(segments)):
                prev_end = segments[i-1].get("end", 0)
                curr_start = segments[i].get("start", 0)
                silence = curr_start - prev_end
                silence_durations.append(silence)
                
                if silence > 5.0:
                    very_long_silences += 1
                elif silence > 2.0:
                    long_silences += 1
            
            avg_silence = sum(silence_durations) / len(silence_durations) if silence_durations else 0
            
            # 2. 발화 길이 분석
            segment_durations = []
            for segment in segments:
                duration = segment.get("end", 0) - segment.get("start", 0)
                segment_durations.append(duration)
            
            avg_duration = sum(segment_durations) / len(segment_durations)
            max_duration = max(segment_durations) if segment_durations else 0
            
            # 3. 텍스트 패턴 분석
            total_text_length = sum(len(segment.get("text", "")) for segment in segments)
            avg_text_length = total_text_length / len(segments)
            
            # 4. 화자 변경 신호 분석
            speaker_change_signals = 0
            for i in range(1, len(segments)):
                curr_text = segments[i].get("text", "").strip()
                prev_text = segments[i-1].get("text", "").strip()
                
                # 대화 신호 키워드 (질문-응답 패턴)
                question_words = ['?', '？', '뭐', '무엇', 'なに', 'what', 'how']
                response_words = ['네', '예', '아니', 'はい', 'そう', 'yes', 'no']
                
                if (any(q in prev_text for q in question_words) and 
                    any(r in curr_text for r in response_words)):
                    speaker_change_signals += 1
            
            # 5. 독백 판단 기준 (훨씬 엄격하게)
            #    → 여러 세그먼트가 오가고, 질문/응답 패턴이 조금이라도 보이면 대화로 처리
            monologue_indicators = [
                very_long_silences == 0,               # 매우 긴 침묵(5초+)이 없음
                avg_silence < 1.0,                     # 평균 침묵이 1초 미만으로 매우 촘촘하게 이어짐
                avg_duration > 3.0 or max_duration > 8.0,  # 발화가 길고 설명 위주인 경우
                avg_text_length > 40,                  # 평균 텍스트가 상당히 길 때만
                speaker_change_signals == 0,           # 질문-응답 패턴이 전혀 없을 때만
                len(segments) <= 3                     # 세그먼트가 아주 적을 때만
            ]

            monologue_score = sum(monologue_indicators)

            logger.info(
                f"독백 분석 결과: 점수 {monologue_score}/6 "
                f"(매우긴침묵: {very_long_silences}, 긴침묵: {long_silences}, "
                f"평균침묵: {avg_silence:.1f}초, 평균발화: {avg_duration:.1f}초, "
                f"최대발화: {max_duration:.1f}초, 평균텍스트: {avg_text_length:.1f}자, "
                f"화자변경신호: {speaker_change_signals})"
            )

            # 이제는 6개 중 5개 이상 만족할 때만 독백으로 본다
            is_monologue = monologue_score >= 5

            if is_monologue:
                logger.info("🎤 독백으로 판단됨 - 독백 전용 처리 모드 활성화")
            else:
                logger.info("💬 대화로 판단됨 - 음성 특성 기반 화자분리 진행")

            return is_monologue
            
        except Exception as e:
            logger.error(f"독백 패턴 감지 실패: {e}")
            # 오류 시 안전하게 독백으로 처리 (단일 화자 우선)
            logger.info("오류로 인해 독백으로 처리")
            return True
    
    def _handle_monologue_segments(self, segments):
        """독백 세그먼트 전용 처리 - 단일 화자 유지"""
        try:
            logger.info("🎤 독백 모드: 단일 화자로 처리")
            
            # 독백은 기본적으로 단일 화자 (화자A)
            speaker_assignments = ["화자A" for _ in segments]
            
            # 독백 내에서도 명확한 주제 전환이 있는 경우에만 화자 분리
            topic_changes = self._detect_strong_topic_changes(segments)
            
            if topic_changes:
                logger.info(f"독백 내 강한 주제 전환 감지: {len(topic_changes)}개 지점")
                current_speaker = 'A'
                
                for i, segment in enumerate(segments):
                    if i in topic_changes:
                        current_speaker = 'B' if current_speaker == 'A' else 'A'
                        logger.info(f"세그먼트 {i}: 강한 주제 전환 → 화자{current_speaker}")
                    
                    speaker_assignments[i] = f"화자{current_speaker}"
            else:
                logger.info("주제 전환 없음 - 완전한 단일 화자 유지")
            
            # 독백 결과 로깅
            from collections import Counter
            speaker_count = Counter(speaker_assignments)
            logger.info(f"🎤 독백 처리 완료: {dict(speaker_count)}")
            
            return speaker_assignments
            
        except Exception as e:
            logger.error(f"독백 세그먼트 처리 실패: {e}")
            return ["화자A" for _ in segments]
    
    def _detect_strong_topic_changes(self, segments):
        """독백 내 강한 주제 전환만 감지 (매우 엄격한 기준)"""
        try:
            topic_changes = []
            
            # 매우 강한 주제 전환 신호만 감지
            strong_transition_keywords = [
                # 한국어 - 명확한 전환
                '그런데 말이야', '아 그리고', '참 그런데', '아 맞다', '그건 그렇고',
                # 일본어 - 명확한 전환  
                'ところで', 'そういえば', 'あ、そうそう', 'それはそうと',
                # 영어 - 명확한 전환
                'by the way', 'speaking of', 'oh and', 'that reminds me'
            ]
            
            for i in range(1, len(segments)):
                curr_text = segments[i].get("text", "").strip().lower()
                
                # 강한 전환 키워드가 있는 경우만
                if any(keyword in curr_text for keyword in strong_transition_keywords):
                    topic_changes.append(i)
                    logger.debug(f"강한 주제 전환 감지: 세그먼트 {i}")
            
            logger.info(f"강한 주제 전환 지점: {len(topic_changes)}개")
            return topic_changes
            
        except Exception as e:
            logger.error(f"강한 주제 전환 감지 실패: {e}")
            return []
    
    def _detect_topic_changes(self, segments):
        """텍스트 기반 주제 변화 감지"""
        try:
            topic_changes = []
            
            # 간단한 키워드 기반 주제 변화 감지
            for i in range(1, len(segments)):
                curr_text = segments[i].get("text", "").strip()
                prev_text = segments[i-1].get("text", "").strip()
                
                # 주제 변화 신호 키워드 (한국어, 일본어, 영어)
                topic_change_keywords = [
                    # 한국어
                    '그런데', '그리고', '또한', '한편', '그래서', '따라서', '결국', '마지막으로',
                    '첫째', '둘째', '셋째', '다음으로', '이제', '그럼', '그러면',
                    # 일본어  
                    'それで', 'そして', 'また', 'しかし', 'でも', 'ところで', 'さて',
                    'まず', '次に', '最後に', '結局', 'つまり', 'だから',
                    # 영어
                    'however', 'but', 'and', 'also', 'then', 'next', 'finally',
                    'first', 'second', 'third', 'so', 'therefore', 'meanwhile'
                ]
                
                # 현재 텍스트에 주제 변화 키워드가 있는지 확인
                curr_lower = curr_text.lower()
                if any(keyword in curr_lower for keyword in topic_change_keywords):
                    topic_changes.append(i)
                    logger.debug(f"주제 변화 감지: 세그먼트 {i} - {curr_text[:30]}...")
                
                # 텍스트 길이 급변 (긴 설명 후 짧은 요약 등)
                if len(prev_text) > 50 and len(curr_text) < 20:
                    topic_changes.append(i)
                    logger.debug(f"텍스트 길이 급변 감지: 세그먼트 {i}")
            
            logger.info(f"주제 변화 지점: {len(topic_changes)}개 - {topic_changes}")
            return topic_changes
            
        except Exception as e:
            logger.error(f"주제 변화 감지 실패: {e}")
            return []
    
    def _extract_voice_features(self, audio_segment, sample_rate):
        """세그먼트에서 음성 특성 추출 (피치, 스펙트럼 중심, MFCC)"""
        try:
            import librosa
            import numpy as np
            
            # numpy 배열로 변환
            if isinstance(audio_segment, torch.Tensor):
                audio_np = audio_segment.numpy()
            else:
                audio_np = audio_segment
            
            # 1. 기본 주파수 (F0) - 피치
            f0 = librosa.yin(audio_np, fmin=50, fmax=400, sr=sample_rate)
            f0_mean = np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 150
            f0_std = np.nanstd(f0[f0 > 0]) if np.any(f0 > 0) else 0
            
            # 2. MFCC (Mel-frequency cepstral coefficients) - 음색 특성
            mfcc = librosa.feature.mfcc(y=audio_np, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 3. 스펙트럴 중심 (Spectral Centroid) - 음성의 밝기
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)
            sc_mean = np.mean(spectral_centroid)
            sc_std = np.std(spectral_centroid)
            
            # 4. 스펙트럴 대역폭 (Spectral Bandwidth)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_np, sr=sample_rate)
            sb_mean = np.mean(spectral_bandwidth)
            
            # 5. 영교차율 (Zero Crossing Rate) - 음성의 거칠기
            zcr = librosa.feature.zero_crossing_rate(audio_np)
            zcr_mean = np.mean(zcr)
            
            # 특성 벡터 구성
            features = np.concatenate([
                [f0_mean, f0_std],           # 피치 특성 (2차원)
                mfcc_mean[:8],               # MFCC 평균 (8차원)
                [sc_mean, sc_std],           # 스펙트럴 중심 (2차원)
                [sb_mean],                   # 스펙트럴 대역폭 (1차원)
                [zcr_mean]                   # 영교차율 (1차원)
            ])
            
            # NaN 값 처리
            features = np.nan_to_num(features, nan=0.0)
            
            logger.debug(f"음성 특성 추출 완료: F0={f0_mean:.1f}Hz, SC={sc_mean:.1f}Hz")
            return features
            
        except ImportError:
            logger.warning("librosa가 설치되지 않아 간단한 특성만 추출")
            return self._extract_simple_voice_features(audio_segment, sample_rate)
        except Exception as e:
            logger.error(f"음성 특성 추출 실패: {e}")
            return None
    
    def _extract_simple_voice_features(self, audio_segment, sample_rate):
        """librosa 없이 간단한 음성 특성 추출"""
        try:
            import numpy as np
            
            if isinstance(audio_segment, torch.Tensor):
                audio_np = audio_segment.numpy()
            else:
                audio_np = audio_segment
            
            # 1. RMS 에너지 (음량)
            rms_energy = np.sqrt(np.mean(audio_np**2))
            
            # 2. 영교차율 (음성의 거칠기)
            zero_crossings = np.sum(np.diff(np.sign(audio_np)) != 0)
            zcr = zero_crossings / len(audio_np)
            
            # 3. 스펙트럼 분석 (FFT)
            fft = np.fft.fft(audio_np)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(audio_np), 1/sample_rate)[:len(fft)//2]
            
            # 스펙트럴 중심 (가중 평균 주파수)
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0
            
            # 주요 주파수 (최대 에너지 주파수)
            dominant_freq = freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0
            
            # 특성 벡터 구성
            features = np.array([
                rms_energy,
                zcr,
                spectral_centroid,
                dominant_freq,
                np.mean(magnitude),
                np.std(magnitude)
            ])
            
            # NaN 값 처리
            features = np.nan_to_num(features, nan=0.0)
            
            logger.debug(f"간단한 음성 특성 추출: 주파수={dominant_freq:.1f}Hz, 에너지={rms_energy:.3f}")
            return features
        except Exception as e:
            logger.error(f"간단한 음성 특성 추출 실패: {e}")
            return None
    
    def _cluster_voice_features(self, voice_features):
        """음성 특성 기반 클러스터링"""
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            features_array = np.array(voice_features)

            # 특성 정규화
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)

            # 최적 클러스터 수 결정 - 더 보수적으로 설정
            n_segments = len(voice_features)
            if n_segments <= 2:
                n_clusters = 1  # 단일 화자
            elif n_segments <= 4:
                n_clusters = 2  # 2명 화자
            elif n_segments <= 7:
                n_clusters = min(2, n_segments - 1)  # 최대 2명
            else:
                n_clusters = min(3, n_segments // 2)  # 기본 최대 3명

            # 전역 설정에 따른 상한 적용 (1~max_speakers)
            max_speakers = getattr(self, "max_speakers", None)
            if max_speakers is not None and max_speakers > 0:
                n_clusters = max(1, min(n_clusters, int(max_speakers)))

            if n_clusters == 1:
                logger.info("단일 화자로 클러스터링")
                return [0] * len(voice_features)

            # K-means 클러스터링 (여러 번 시도해서 최적 결과 선택)
            best_labels = None
            best_inertia = float('inf')

            for attempt in range(5):  # 5번 시도
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42 + attempt, n_init=10)
                    labels = kmeans.fit_predict(features_normalized)

                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        best_labels = labels
                        best_centers = kmeans.cluster_centers_
                except Exception:
                    continue

            if best_labels is None:
                logger.warning("클러스터링 실패 - 단일 화자로 처리")
                return [0] * len(voice_features)

            logger.info(f"음성 특성 클러스터링 완료: {n_clusters}명 화자 감지 (관성: {best_inertia:.2f})")

            # 클러스터 품질 검증
            unique_labels = len(set(best_labels))
            if unique_labels < n_clusters:
                logger.warning(f"일부 클러스터가 비어있음: {unique_labels}/{n_clusters}")

            # 클러스터 중심점 정보 로깅 (간단 요약)
            for i, center in enumerate(best_centers):
                logger.debug(f"화자{chr(65 + i)} 특성 요약: F0={center[0]:.1f}, MFCC1={center[2]:.2f}")

            return best_labels

        except ImportError:
            logger.warning("scikit-learn이 설치되지 않아 간단한 분류 사용")
            return self._simple_voice_clustering(voice_features)
        except Exception as e:
            logger.error(f"음성 특성 클러스터링 실패: {e}")
            return [0] * len(voice_features)  # 모두 같은 화자로 처리

    def _simple_voice_clustering(self, voice_features):
        """간단한 음성 특성 기반 분류 (scikit-learn 미사용 시)"""
        try:
            import numpy as np

            features_array = np.array(voice_features)

            # 첫 번째 특성(예: 피치)을 기준으로 2그룹 분할
            first_feature = features_array[:, 0]
            median_value = np.median(first_feature)
            labels = (first_feature > median_value).astype(int)

            logger.info(f"간단한 음성 분류 완료: 기준값={median_value:.2f}")
            return labels

        except Exception as e:
            logger.error(f"간단한 음성 분류 실패: {e}")
            return [0] * len(voice_features)

    def _assign_speakers_from_clusters(self, segments, valid_segments, speaker_labels):
        """클러스터링 결과를 전체 세그먼트에 할당"""
        speaker_assignments = []
        label_to_speaker = {}

        # 라벨을 화자 이름으로 매핑
        unique_labels = sorted(set(speaker_labels))
        for i, label in enumerate(unique_labels):
            speaker_letter = chr(ord('A') + i)
            label_to_speaker[label] = f"화자{speaker_letter}"

        # 유효한 세그먼트의 인덱스와 라벨 매핑
        valid_assignments = {}
        for (seg_idx, _segment), label in zip(valid_segments, speaker_labels):
            valid_assignments[seg_idx] = label_to_speaker[label]

        # 전체 세그먼트에 화자 할당 (앞에서 정한 화자 유지)
        current_speaker = "화자A"
        for i, _segment in enumerate(segments):
            if i in valid_assignments:
                current_speaker = valid_assignments[i]
            speaker_assignments.append(current_speaker)

        # 화자 일관성 후처리
        speaker_assignments = self._post_process_voice_consistency(
            speaker_assignments, valid_segments, speaker_labels
        )

        from collections import Counter
        speaker_count = Counter(speaker_assignments)
        logger.info(f"음성 특성 기반 화자 분포: {dict(speaker_count)}")

        return speaker_assignments

    def unload_models(self):
        """로딩된 STT/노이즈제거/화자분리 모델을 메모리에서 해제"""
        try:
            logger.info("🧹 오디오 파이프라인 모델 언로드 시작")

            # Whisper STT 모델 해제
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None

            # 노이즈 제거 모델 해제
            if self.denoiser not in (None, "simple_filter"):
                try:
                    del self.denoiser
                except Exception:
                    pass
                self.denoiser = None

            # 화자 인코더 모델 해제
            if self.speaker_encoder is not None:
                try:
                    del self.speaker_encoder
                except Exception:
                    pass
                self.speaker_encoder = None

            # GPU 메모리 정리
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ 오디오 파이프라인 모델 언로드 완료")
        except Exception as e:
            logger.error(f"오디오 파이프라인 모델 언로드 중 오류: {e}")

    def _post_process_voice_consistency(self, speaker_assignments, valid_segments, speaker_labels):
        """음성 특성 기반 화자 일관성 후처리"""
        try:
            from collections import Counter

            logger.info("음성 특성 기반 일관성 후처리 시작...")

            speaker_counts = Counter(speaker_assignments)
            logger.info(f"후처리 전 화자 분포: {dict(speaker_counts)}")

            # 화자 수가 이미 적으면(<= max_speakers) 보수적으로 유지
            max_speakers = getattr(self, "max_speakers", None)
            if max_speakers is not None and max_speakers > 0:
                if len(speaker_counts) <= max_speakers:
                    logger.info("화자 수가 설정 상한 이내 - 후처리 최소화")
                    return speaker_assignments

            # 고립된 화자(세그먼트 1개만 가진 화자)만 조심스럽게 통합
            isolated_speakers = [s for s, c in speaker_counts.items() if c == 1]
            if not isolated_speakers:
                logger.info("고립된 화자 없음 - 후처리 완료")
                return speaker_assignments

            assignments = list(speaker_assignments)

            for isolated in isolated_speakers:
                idx = assignments.index(isolated)
                prev_spk = assignments[idx - 1] if idx > 0 else None
                next_spk = assignments[idx + 1] if idx < len(assignments) - 1 else None

                target = None
                if prev_spk is not None and prev_spk == next_spk and speaker_counts[prev_spk] >= 3:
                    target = prev_spk

                if target:
                    logger.info(f"고립된 화자 {isolated} → {target} 로 통합")
                    assignments[idx] = target

            final_counts = Counter(assignments)
            logger.info(f"후처리 후 화자 분포: {dict(final_counts)}")
            return assignments

        except Exception as e:
            logger.error(f"음성 특성 일관성 후처리 실패: {e}")
            return speaker_assignments

    def _fallback_speaker_assignment(self, segments):
        """음성 특성 추출 실패 시 대안 로직"""
        logger.info("대안 화자 할당 로직 사용")

        # 1.5초 이상 침묵 기준으로 간단 분리 (A/B 번갈아 가며)
        speaker_assignments = []
        current_speaker = 'A'

        for i, segment in enumerate(segments):
            if i > 0:
                prev_end = segments[i - 1].get("end", 0)
                curr_start = segment.get("start", 0)
                silence_duration = curr_start - prev_end

                if silence_duration > 1.5:
                    current_speaker = 'B' if current_speaker == 'A' else 'A'

            speaker_assignments.append(f"화자{current_speaker}")

        return speaker_assignments

    def _is_single_speaker(self, segments):
        """간단한 단일 화자 판단 로직"""
        if len(segments) <= 2:
            logger.info("세그먼트 2개 이하 - 단일 화자로 판단")
            return True

        long_silence_count = 0
        for i in range(1, len(segments)):
            prev_end = segments[i - 1].get("end", 0)
            curr_start = segments[i].get("start", 0)
            silence_duration = curr_start - prev_end

            if silence_duration > 1.5:
                long_silence_count += 1

        is_single = long_silence_count == 0
        logger.info(
            f"단일 화자 판단: {'단일' if is_single else '다중'} "
            f"(1.5초+ 침묵: {long_silence_count}회, 세그먼트: {len(segments)}개)"
        )

        return is_single

    def _assign_ecapa_speakers(self, audio_file, segments):
        """ECAPA-VOXCELEB 기반 실제 화자분리"""
        # 화자 임베딩 추출
        embedding_result = self._extract_speaker_embeddings(audio_file, segments)
        
        if embedding_result is None:
            logger.warning("ECAPA 화자분리 실패, 규칙 기반 방식 사용")
            return self._assign_smart_speakers(segments)
        
        embeddings, valid_segments = embedding_result
        
        # 화자 클러스터링
        cluster_labels = self._cluster_speakers(embeddings)
        
        if cluster_labels is None:
            logger.warning("화자 클러스터링 실패, 규칙 기반 방식 사용")
            return self._assign_smart_speakers(segments)
        
        # 클러스터 라벨을 화자 이름으로 변환
        unique_labels = sorted(set(cluster_labels))
        label_to_speaker = {}
        
        for i, label in enumerate(unique_labels):
            speaker_letter = chr(ord('A') + i)
            label_to_speaker[label] = f"화자{speaker_letter}"
        
        # 전체 세그먼트에 화자 할당
        speaker_assignments = []
        valid_idx = 0
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                # 빈 텍스트는 이전 화자 유지
                if speaker_assignments:
                    speaker_assignments.append(speaker_assignments[-1])
                else:
                    speaker_assignments.append("화자A")
                continue
            
            # 유효한 세그먼트인지 확인
            if valid_idx < len(valid_segments) and segment == valid_segments[valid_idx]:
                # 클러스터링 결과 사용
                cluster_label = cluster_labels[valid_idx]
                speaker_name = label_to_speaker[cluster_label]
                speaker_assignments.append(speaker_name)
                valid_idx += 1
            else:
                # 유효하지 않은 세그먼트는 이전 화자 유지
                if speaker_assignments:
                    speaker_assignments.append(speaker_assignments[-1])
                else:
                    speaker_assignments.append("화자A")
        
        return speaker_assignments
    
    def process_single_file(self, input_file):
        """
        단일 오디오 파일 전체 파이프라인 처리
        
        Args:
            input_file (str): 입력 오디오 또는 비디오 파일 경로
        """
        try:
            input_path = Path(input_file)
            file_stem = input_path.stem
            
            # 오디오 파일 여부 확인
            if self._is_audio_file(input_file):
                logger.info(f"🎵 오디오 파일 감지: {input_file}")
                audio_file = input_file
            else:
                raise ValueError(f"지원하지 않는 오디오 파일 형식: {input_path.suffix}")
            
            # 1단계: 노이즈 제거
            denoised_file = self.audio_out_dir / f"{file_stem}_denoised.wav"
            self.denoise_audio(audio_file, str(denoised_file))
            
            # 2단계: audio_output으로 복사 (파이프라인 구조 유지)
            output_audio_file = self.audio_output_dir / f"{file_stem}_denoised.wav"
            import shutil
            shutil.copy2(denoised_file, output_audio_file)
            
            # 3단계: STT 처리 (denoised 오디오 사용, 화자분리는 원본 오디오 사용)
            transcript_file = self.script_output_dir / f"{file_stem}_transcript.txt"
            srt_file = self.script_output_dir / f"{file_stem}_subtitle.srt"
            transcribed_text = self.transcribe_audio(
                str(output_audio_file),
                str(transcript_file),
                str(srt_file),
                diarization_audio_file=str(audio_file)
            )
            
            logger.info(f"파일 처리 완료: {input_file}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"파일 처리 실패 ({input_file}): {e}")
            raise

    def transcribe_uploaded_wav(self, wav_path, save_dir=None, create_srt=True, enable_diarization=True, enable_timestamps=True):
        """이미 추출된 WAV 파일을 입력으로 받아 전사 및 화자분리를 수행하는 백엔드용 헬퍼

        Args:
            wav_path (str): FFmpeg 등으로 미리 추출된 단일 채널/16kHz WAV 파일 경로
            save_dir (str, optional): 결과 파일을 저장할 기본 디렉토리.
                None 이면 입력 wav 파일과 같은 디렉토리를 사용.
            create_srt (bool): SRT 자막 파일 생성 여부
            enable_diarization (bool): 화자분리 활성화 여부
            enable_timestamps (bool): 타임스탬프 활성화 여부

        Returns:
            dict: {
                "text": 전체 전사 텍스트 (str),
                "denoised_wav": 노이즈 제거된 wav 경로 (str),
                "transcript_path": 타임스탬프 포함 전사 txt 경로 (str),
                "simple_path": 간단 전사(txt, 화자/타임스탬프 포함) 경로 (str),
                "text_only_path": 텍스트 전용 파일 경로 (str),
                "srt_path": SRT 자막 경로 또는 None,
            }
        """
        try:
            wav_path = Path(wav_path)
            if not wav_path.exists():
                raise FileNotFoundError(f"입력 WAV 파일을 찾을 수 없습니다: {wav_path}")

            base_dir = Path(save_dir) if save_dir is not None else wav_path.parent
            base_dir.mkdir(parents=True, exist_ok=True)

            file_stem = wav_path.stem

            # 1단계: 노이즈 제거 (기존 로직 재사용)
            denoised_file = base_dir / f"{file_stem}_denoised.wav"
            logger.info(f"[backend] 업로드 WAV 노이즈 제거 시작: {wav_path} -> {denoised_file}")
            self.denoise_audio(str(wav_path), str(denoised_file))

            # 2단계: STT + 화자분리 (STT는 denoised, 화자분리는 원본 wav 사용)
            transcript_file = base_dir / f"{file_stem}_transcript.txt"
            srt_file = base_dir / f"{file_stem}_subtitle.srt" if create_srt else None

            logger.info(f"[backend] 업로드 WAV STT 처리 시작: {denoised_file}")
            text = self.transcribe_audio(
                str(denoised_file),
                str(transcript_file),
                str(srt_file) if srt_file else None,
                diarization_audio_file=str(wav_path) if enable_diarization else None,
                enable_timestamps=enable_timestamps
            )

            # transcribe_audio 내부에서 simple 텍스트도 생성됨
            simple_file = base_dir / f"{file_stem}_transcript_simple.txt"

            # 백엔드에서 바로 보기 좋은 순수 텍스트 전용 파일 생성
            text_only_file = base_dir / f"{file_stem}_text_only.txt"
            try:
                with text_only_file.open("w", encoding="utf-8") as f:
                    f.write(text.strip() + "\n")
                logger.info(f"[backend] 텍스트 전용 파일 생성: {text_only_file}")
            except Exception as e:
                logger.error(f"텍스트 전용 파일 생성 실패: {e}")

            result = {
                "text": text,
                "denoised_wav": str(denoised_file),
                "transcript_path": str(transcript_file),
                "simple_path": str(simple_file),
                "text_only_path": str(text_only_file),
                "srt_path": str(srt_file) if srt_file else None,
            }

            logger.info(f"[backend] 업로드 WAV 처리 완료: {wav_path}")
            return result

        except Exception as e:
            logger.error(f"업로드 WAV 전사 처리 실패 ({wav_path}): {e}")
            raise
    
    def process_all_files(self):
        """audio_input 폴더의 모든 오디오 파일 처리"""
        # 지원하는 오디오 파일 확장자
        file_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg', '*.aac']
        
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(glob.glob(str(self.audio_input_dir / ext)))
        
        if not audio_files:
            logger.warning("audio_input 폴더에 처리할 오디오 파일이 없습니다.")
            logger.info(f"지원 형식: {', '.join(file_extensions)}")
            return
        
        logger.info(f"총 {len(audio_files)}개 오디오 파일 처리 시작")
        
        results = []
        for i, media_file in enumerate(audio_files, 1):
            try:
                logger.info(f"[{i}/{len(audio_files)}] 🎵 오디오 처리 중: {Path(media_file).name}")
                result = self.process_single_file(media_file)
                results.append({
                    'file': Path(media_file).name,
                    'type': 'audio',
                    'status': 'success',
                    'text': result
                })
            except Exception as e:
                logger.error(f"파일 처리 실패: {Path(media_file).name} - {e}")
                results.append({
                    'file': Path(media_file).name,
                    'type': 'audio',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # 처리 결과 요약 저장
        self._save_processing_summary(results)
        
        logger.info("모든 파일 처리 완료")
        return results
    
    def _save_processing_summary(self, results):
        """처리 결과 요약 저장"""
        summary_file = self.script_output_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 오디오 파이프라인 처리 결과 요약 ===\n")
            f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 파일 수: {len(results)}\n")
            
            success_count = sum(1 for r in results if r['status'] == 'success')
            failed_count = len(results) - success_count
            
            f.write(f"성공: {success_count}개\n")
            f.write(f"실패: {failed_count}개\n")
            f.write("-" * 50 + "\n\n")
            
            for result in results:
                f.write(f"파일: {result['file']}\n")
                f.write(f"상태: {result['status']}\n")
                if result['status'] == 'success':
                    f.write(f"텍스트: {result['text'][:200]}...\n")
                else:
                    f.write(f"오류: {result['error']}\n")
                f.write("-" * 30 + "\n")
        
        logger.info(f"처리 결과 요약 저장: {summary_file}")

def main(target_language=None):
    """
    메인 함수
    
    Args:
        target_language (str): 대상 언어 코드 ('ko', 'ja', 'en', etc.)
    """
    print("=== 음성 및 비디오 파이프라인 시작 ===")
    print("🎵 오디오: audio_input → 노이즈제거 → audio_out → STT → script_output")
    print("🎬 비디오: video_input → 오디오추출 → 노이즈제거 → STT → script_output")
    print()
    
    # 언어 설정 안내
    if target_language:
        supported_languages = {
            'ko': '한국어', 'ja': '日本語', 'en': 'English', 'zh': '中文',
            'es': 'Español', 'fr': 'Français', 'de': 'Deutsch', 'ru': 'Русский'
        }
        lang_name = supported_languages.get(target_language, target_language)
        print(f"🌐 대상 언어: {lang_name} ({target_language})")
    else:
        print("🌐 언어: 자동 감지 모드")
    print()
    
    try:
        # GPU 사용 가능 여부 확인
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  GPU 사용 불가 - CPU 모드로 실행")
        
        # 파이프라인 초기화
        pipeline = AudioPipeline(use_gpu=use_gpu, target_language=target_language)
        
        # 모든 파일 처리
        results = pipeline.process_all_files()
        
        # 결과 출력
        if results:
            success_count = sum(1 for r in results if r['status'] == 'success')
            print(f"\n🎉 처리 완료: {success_count}/{len(results)}개 파일 성공")
        else:
            print("\n⚠️  처리할 파일이 없습니다.")
            print("audio_input 폴더에 오디오 파일을 넣어주세요.")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# 사용 예시:
# 
# 1. 오디오 파일 처리:
#    audio_input/ 폴더에 WAV, MP3, M4A, FLAC, OGG 파일 저장 후 실행
#
# 2. 비디오 파일 처리 (FFmpeg 필요):
#    audio_input/ 폴더에 MP4, AVI, MOV, MKV, WEBM 파일 저장 후 실행
#    
# 3. 특정 언어 지정:
#    pipeline = AudioPipeline(target_language='ko')  # 한국어
#    pipeline = AudioPipeline(target_language='ja')  # 일본어
#
# 4. 단일 파일 처리:
#    pipeline = AudioPipeline()
#    result = pipeline.process_single_file("video.mp4")
#
# 5. FFmpeg 설치 확인:
#    pipeline = AudioPipeline()
#    print(f"FFmpeg 사용 가능: {pipeline.ffmpeg_available}")
#
# 출력 파일:
# - audio_out/: 노이즈 제거된 오디오
# - script_output/: 전사 텍스트 및 SRT 자막
# - processing_summary_*.txt: 배치 처리 결과 요약
"""
audio_pipeline.py 파일 맨 아래에 추가할 코드
(1610줄 이후에 추가)
"""

# ===== 전역 인스턴스 및 초기화 함수 추가 =====

# 전역 인스턴스
audio_pipeline_instance = None


def initialize_audio_pipeline(
    use_gpu: bool = True,
    target_language: Optional[str] = None,
    whisper_model_size: str = "large-v3",
    load_denoiser: bool = True,
    load_speaker_encoder: bool = True
):
    """
    오디오 파이프라인 초기화 및 모델 로드
    
    Args:
        use_gpu: GPU 사용 여부
        target_language: 대상 언어 (None=자동감지)
        whisper_model_size: Whisper 모델 크기
        load_denoiser: 노이즈 제거 모델 로드 여부
        load_speaker_encoder: 화자분리 모델 로드 여부
    """
    global audio_pipeline_instance
    
    logger.info("="*60)
    logger.info("🚀 오디오 파이프라인 초기화 시작")
    logger.info("="*60)
    
    try:
        # GPU 확인
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU를 사용할 수 없습니다. CPU 모드로 전환합니다.")
            use_gpu = False
        
        if use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU: {gpu_name} ({total_memory:.1f}GB)")
        else:
            logger.info("⚠️  CPU 모드")
        
        # AudioPipeline 인스턴스 생성
        logger.info("\n[1/4] AudioPipeline 인스턴스 생성...")
        audio_pipeline_instance = AudioPipeline(
            use_gpu=use_gpu,
            target_language=target_language
        )
        logger.info(f"   디바이스: {audio_pipeline_instance.device}")
        
        # Whisper 모델 로드
        logger.info(f"\n[2/4] Whisper 모델 로딩 ({whisper_model_size})...")
        audio_pipeline_instance._load_whisper(model_size=whisper_model_size)
        logger.info("✅ Whisper 로딩 완료")
        
        if use_gpu:
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"   GPU 메모리 사용: {allocated:.2f}GB")
        
        # 노이즈 제거 모델 로드
        if load_denoiser:
            logger.info("\n[3/4] 노이즈 제거 모델 로딩...")
            try:
                audio_pipeline_instance._load_denoiser()
                logger.info("✅ 노이즈 제거 모델 로딩 완료")
            except Exception as e:
                logger.warning(f"⚠️  노이즈 제거 모델 로딩 실패: {e}")
        else:
            logger.info("\n[3/4] 노이즈 제거 모델 스킵")
        
        # 화자분리 모델 로드
        if load_speaker_encoder:
            logger.info("\n[4/4] 화자분리 모델 로딩...")
            try:
                success = audio_pipeline_instance._load_speaker_encoder()
                if success:
                    logger.info("✅ 화자분리 모델 로딩 완료")
                else:
                    logger.warning("⚠️  화자분리 모델 로딩 실패")
            except Exception as e:
                logger.warning(f"⚠️  화자분리 모델 로딩 실패: {e}")
        else:
            logger.info("\n[4/4] 화자분리 모델 스킵")
        
        logger.info("\n" + "="*60)
        logger.info("✅ 오디오 파이프라인 초기화 완료!")
        logger.info("="*60)
        
        return audio_pipeline_instance
        
    except Exception as e:
        logger.error(f"❌ 오디오 파이프라인 초기화 실패: {e}")
        audio_pipeline_instance = None
        raise


def get_pipeline_status() -> dict:
    """현재 파이프라인 상태 조회"""
    if audio_pipeline_instance is None:
        return {
            "initialized": False,
            "error": "파이프라인이 초기화되지 않았습니다."
        }
    
    status = {
        "initialized": True,
        "device": audio_pipeline_instance.device,
        "use_gpu": audio_pipeline_instance.use_gpu,
        "target_language": audio_pipeline_instance.target_language or "auto",
        "models": {
            "whisper": audio_pipeline_instance.whisper_model is not None,
            "denoiser": audio_pipeline_instance.denoiser is not None,
            "speaker_encoder": audio_pipeline_instance.speaker_encoder is not None
        }
    }
    
    if audio_pipeline_instance.use_gpu and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        status["gpu_memory"] = {
            "allocated_gb": round(allocated, 2),
            "total_gb": round(total, 2),
            "usage_percent": round((allocated / total) * 100, 1)
        }
    
    return status


def get_memory_stats() -> dict:
    """GPU 메모리 상태 조회"""
    if audio_pipeline_instance is None or not audio_pipeline_instance.use_gpu:
        return {"error": "GPU를 사용하지 않거나 파이프라인이 초기화되지 않았습니다."}
    
    if not torch.cuda.is_available():
        return {"error": "GPU를 사용할 수 없습니다."}
    