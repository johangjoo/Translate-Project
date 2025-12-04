"""
STT (Speech-to-Text) 추론 모듈 - API 전용
원본: audio_pipeline.py에서 핵심 기능만 추출
"""

import torch
import torchaudio
import whisper
import logging
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WhisperSTT:
    """
    Whisper 기반 음성인식 클래스 (API 최적화 버전)
    """
    
    def __init__(
        self, 
        model_size: str = "medium",
        use_gpu: bool = True,
        language: Optional[str] = None
    ):
        """
        초기화
        
        Args:
            model_size: Whisper 모델 크기 ('tiny', 'base', 'small', 'medium', 'large', 'large-v3')
            use_gpu: GPU 사용 여부
            language: 타겟 언어 코드 (None이면 자동 감지)
                     'en', 'ko', 'ja', 'zh', 'es', 'fr', 'de', 'ru' 등
        """
        self.model_size = model_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.language = language
        self.model = None
        
        logger.info(f"WhisperSTT 초기화 - 디바이스: {self.device}, 모델: {model_size}")
        
        # 지원 언어 정보
        self.supported_languages = {
            'en': 'English',
            'ko': '한국어',
            'ja': '日本語',
            'zh': '中文',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'ru': 'Русский'
        }
    
    def load_model(self):
        """Whisper 모델 로딩 (서버 시작 시 1회 호출)"""
        if self.model is None:
            try:
                logger.info(f"Whisper 모델 ({self.model_size}) 로딩 중...")
                self.model = whisper.load_model(self.model_size, device=self.device)
                logger.info("✅ Whisper 모델 로딩 완료")
            except Exception as e:
                logger.error(f"❌ Whisper 모델 로딩 실패: {e}")
                raise
    
    def transcribe(
        self, 
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        음성 파일을 텍스트로 변환
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 지정 (None이면 자동 감지 또는 초기화 시 설정한 언어)
            word_timestamps: 단어별 타임스탬프 포함 여부
            initial_prompt: 초기 프롬프트 (컨텍스트 제공)
        
        Returns:
            {
                "text": str,  # 전체 텍스트
                "segments": [...],  # 세그먼트별 정보
                "language": str  # 감지/사용된 언어
            }
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        try:
            logger.info(f"🎤 STT 처리 시작: {audio_path}")
            
            # 언어 설정 우선순위: 메서드 파라미터 > 인스턴스 설정 > 자동 감지
            target_lang = language or self.language
            
            # Whisper transcribe 옵션
            transcribe_options = {
                "word_timestamps": word_timestamps,
                "verbose": False
            }
            
            if target_lang:
                transcribe_options["language"] = target_lang
                lang_name = self.supported_languages.get(target_lang, target_lang)
                logger.info(f"언어 지정: {lang_name}")
            else:
                logger.info("언어 자동 감지 모드")
            
            if initial_prompt:
                transcribe_options["initial_prompt"] = initial_prompt
            
            # STT 수행
            result = self.model.transcribe(str(audio_path), **transcribe_options)
            
            transcribed_text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            logger.info(f"✅ STT 완료 - 언어: {detected_language}")
            logger.info(f"📝 텍스트: {transcribed_text[:100]}...")
            
            return {
                "text": transcribed_text,
                "segments": result.get("segments", []),
                "language": detected_language
            }
            
        except Exception as e:
            logger.error(f"❌ STT 처리 실패: {e}")
            raise
    
    def transcribe_simple(
        self, 
        audio_path: str,
        language: Optional[str] = None
    ) -> str:
        """
        간단한 버전 - 텍스트만 반환
        
        Args:
            audio_path: 오디오 파일 경로
            language: 언어 지정 (선택사항)
        
        Returns:
            str: 변환된 텍스트
        """
        result = self.transcribe(audio_path, language=language)
        return result["text"]
    
    def unload_model(self):
        """메모리 해제"""
        if self.model is not None:
            try:
                logger.info("🔄 Whisper 모델 GPU 메모리 해제 중...")
                
                # GPU에서 CPU로 이동 (GPU 메모리 확보)
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')
                
                # 모델의 모든 파라미터를 CPU로 명시적으로 이동
                if hasattr(self.model, 'parameters'):
                    for param in self.model.parameters():
                        if param.is_cuda:
                            param.data = param.data.cpu()
                
                # 모델의 모든 버퍼를 CPU로 이동
                if hasattr(self.model, 'buffers'):
                    for buffer in self.model.buffers():
                        if buffer.is_cuda:
                            buffer.data = buffer.data.cpu()
                
                # 모델 삭제
                del self.model
                self.model = None
                
                # 가비지 컬렉션 실행 (여러 번 실행하여 순환 참조 정리)
                import gc
                gc.collect()
                gc.collect()
                gc.collect()  # 세 번째로 확실하게 정리
                
                # GPU 메모리 정리 (더 강력하게)
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()  # 한 번 더
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        # CUDA IPC 메모리 정리 (공유 메모리)
                        if hasattr(torch.cuda, 'ipc_collect'):
                            torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    
                    # 현재 GPU 메모리 사용량 로깅
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    logger.info(f"✅ Whisper 모델 언로드 완료 (GPU 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB)")
                else:
                    logger.info("✅ Whisper 모델 언로드 완료")
            except Exception as e:
                logger.warning(f"Whisper 모델 언로드 중 오류 (무시): {e}")
                self.model = None


class AudioDenoiser:
    """
    간단한 오디오 노이즈 제거 클래스 (선택사항)
    SpeechBrain 없이도 작동하는 간단한 필터
    """
    
    def __init__(self):
        self.target_sr = 16000
        logger.info("AudioDenoiser 초기화 (간단한 필터링)")
    
    def denoise(self, input_path: str, output_path: str):
        """
        노이즈 제거 수행
        
        Args:
            input_path: 입력 오디오 경로
            output_path: 출력 오디오 경로
        """
        try:
            logger.info(f"🔧 노이즈 제거 시작: {input_path}")
            
            # 오디오 로드
            waveform, sample_rate = torchaudio.load(input_path)
            
            # 모노로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 16kHz로 리샘플링
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
                waveform = resampler(waveform)
                sample_rate = self.target_sr
            
            # 간단한 필터링 적용
            filtered_waveform = self._apply_simple_filter(waveform.squeeze(0), sample_rate)
            filtered_waveform = filtered_waveform.unsqueeze(0)
            
            # 저장
            torchaudio.save(output_path, filtered_waveform, sample_rate)
            
            logger.info(f"✅ 노이즈 제거 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ 노이즈 제거 실패: {e}")
            raise
    
    def _apply_simple_filter(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        간단한 스펙트럼 필터링 적용
        
        Args:
            waveform: 오디오 파형
            sample_rate: 샘플링 레이트
        
        Returns:
            필터링된 파형
        """
        try:
            from scipy.signal import butter, filtfilt
            
            # 80Hz 이하 저주파 노이즈 제거 (고역 통과 필터)
            nyquist = sample_rate / 2
            low_cutoff = 80 / nyquist
            b, a = butter(4, low_cutoff, btype='high')
            
            # 필터 적용
            filtered = filtfilt(b, a, waveform.numpy())
            
            # 정규화 (0.8배로 안전 마진)
            filtered = filtered / np.max(np.abs(filtered)) * 0.8
            
            return torch.from_numpy(filtered).float()
            
        except ImportError:
            logger.warning("scipy 미설치 - 간단한 정규화만 적용")
            # scipy 없으면 간단한 정규화만
            normalized = waveform / torch.max(torch.abs(waveform)) * 0.8
            return normalized


# ===== API에서 사용할 전역 인스턴스 =====
whisper_stt: Optional[WhisperSTT] = None
audio_denoiser: Optional[AudioDenoiser] = None


def initialize_stt_models(
    whisper_model_size: str = "medium",
    language: Optional[str] = None,
    use_denoiser: bool = False
):
    """
    STT 모델 초기화 (서버 시작 시 호출)
    
    Args:
        whisper_model_size: Whisper 모델 크기
        language: 기본 언어 설정
        use_denoiser: 노이즈 제거 사용 여부
    """
    global whisper_stt, audio_denoiser
    
    logger.info("="*50)
    logger.info("🚀 STT 모델 초기화 시작...")
    logger.info("="*50)
    
    # Whisper STT 초기화
    whisper_stt = WhisperSTT(
        model_size=whisper_model_size,
        use_gpu=True,
        language=language
    )
    whisper_stt.load_model()
    
    # 노이즈 제거 (선택사항)
    if use_denoiser:
        audio_denoiser = AudioDenoiser()
        logger.info("✅ 노이즈 제거 활성화")
    
    logger.info("="*50)
    logger.info("✅ STT 모델 초기화 완료!")
    logger.info("="*50)


class TranslateModel:
    def __init__(self, base_model_path, lora_path, s_lang, t_lang):
        pass
    def load_model(self):
        pass
    def translate(self, text: str, source_lang='s', target_lang='t', max_length=512):
        pass
    def unload_model(self):
        pass
    pass

def initialize_models(config):
    pass