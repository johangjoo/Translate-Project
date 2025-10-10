import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
import os
from rnnaudio_huggingface import HuggingFaceDenoiser, apply_denoising_huggingface

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class AudioFrequencyVisualizer:
    """오디오 주파수 분석 및 시각화 클래스"""
    
    def __init__(self, sr=16000):
        """
        초기화
        
        Args:
            sr (int): 샘플링 레이트
        """
        self.sr = sr
        
    def load_audio(self, audio_path):
        """
        오디오 파일 로드
        
        Args:
            audio_path (str): 오디오 파일 경로
            
        Returns:
            tuple: (오디오 데이터, 샘플링 레이트)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            return y, sr
        except Exception as e:
            print(f"오디오 로드 실패: {e}")
            return None, None
    
    def plot_waveform_comparison(self, original_path, denoised_path, save_path=None):
        """
        원본과 노이즈 제거된 오디오의 파형 비교
        
        Args:
            original_path (str): 원본 오디오 파일 경로
            denoised_path (str): 노이즈 제거된 오디오 파일 경로
            save_path (str): 저장할 이미지 경로 (선택사항)
        """
        # 오디오 로드
        y_orig, sr_orig = self.load_audio(original_path)
        y_clean, sr_clean = self.load_audio(denoised_path)
        
        if y_orig is None or y_clean is None:
            print("오디오 파일을 로드할 수 없습니다.")
            return
        
        # 시간 축 생성
        t_orig = np.linspace(0, len(y_orig)/sr_orig, len(y_orig))
        t_clean = np.linspace(0, len(y_clean)/sr_clean, len(y_clean))
        
        # 플롯 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 원본 파형
        ax1.plot(t_orig, y_orig, color='red', alpha=0.7)
        ax1.set_title('원본 오디오 (노이즈 포함)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('시간 (초)')
        ax1.set_ylabel('진폭')
        ax1.grid(True, alpha=0.3)
        
        # 노이즈 제거된 파형
        ax2.plot(t_clean, y_clean, color='blue', alpha=0.7)
        ax2.set_title('노이즈 제거된 오디오', fontsize=14, fontweight='bold')
        ax2.set_xlabel('시간 (초)')
        ax2.set_ylabel('진폭')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"파형 비교 이미지 저장: {save_path}")
        
        plt.show()
    
    def plot_spectrogram_comparison(self, original_path, denoised_path, save_path=None):
        """
        원본과 노이즈 제거된 오디오의 스펙트로그램 비교
        
        Args:
            original_path (str): 원본 오디오 파일 경로
            denoised_path (str): 노이즈 제거된 오디오 파일 경로
            save_path (str): 저장할 이미지 경로 (선택사항)
        """
        # 오디오 로드
        y_orig, sr_orig = self.load_audio(original_path)
        y_clean, sr_clean = self.load_audio(denoised_path)
        
        if y_orig is None or y_clean is None:
            print("오디오 파일을 로드할 수 없습니다.")
            return
        
        # 스펙트로그램 계산
        D_orig = librosa.stft(y_orig)
        D_clean = librosa.stft(y_clean)
        
        # dB 스케일로 변환
        DB_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
        DB_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
        
        # 플롯 생성
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # 원본 스펙트로그램
        img1 = librosa.display.specshow(DB_orig, sr=sr_orig, x_axis='time', y_axis='hz', ax=ax1)
        ax1.set_title('원본 오디오 스펙트로그램 (노이즈 포함)', fontsize=14, fontweight='bold')
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # 노이즈 제거된 스펙트로그램
        img2 = librosa.display.specshow(DB_clean, sr=sr_clean, x_axis='time', y_axis='hz', ax=ax2)
        ax2.set_title('노이즈 제거된 오디오 스펙트로그램', fontsize=14, fontweight='bold')
        fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
        
        # 차이 스펙트로그램
        # 길이를 맞춤
        min_len = min(DB_orig.shape[1], DB_clean.shape[1])
        diff_spec = DB_orig[:, :min_len] - DB_clean[:, :min_len]
        
        img3 = librosa.display.specshow(diff_spec, sr=sr_orig, x_axis='time', y_axis='hz', ax=ax3, cmap='RdBu_r')
        ax3.set_title('스펙트로그램 차이 (제거된 노이즈)', fontsize=14, fontweight='bold')
        fig.colorbar(img3, ax=ax3, format='%+2.0f dB')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"스펙트로그램 비교 이미지 저장: {save_path}")
        
        plt.show()
    
    def plot_frequency_domain_comparison(self, original_path, denoised_path, save_path=None):
        """
        주파수 도메인에서의 비교 분석
        
        Args:
            original_path (str): 원본 오디오 파일 경로
            denoised_path (str): 노이즈 제거된 오디오 파일 경로
            save_path (str): 저장할 이미지 경로 (선택사항)
        """
        # 오디오 로드
        y_orig, sr_orig = self.load_audio(original_path)
        y_clean, sr_clean = self.load_audio(denoised_path)
        
        if y_orig is None or y_clean is None:
            print("오디오 파일을 로드할 수 없습니다.")
            return
        
        # FFT 계산
        fft_orig = np.fft.fft(y_orig)
        fft_clean = np.fft.fft(y_clean)
        
        # 주파수 축 생성
        freqs_orig = np.fft.fftfreq(len(fft_orig), 1/sr_orig)
        freqs_clean = np.fft.fftfreq(len(fft_clean), 1/sr_clean)
        
        # 양의 주파수만 사용
        pos_mask_orig = freqs_orig >= 0
        pos_mask_clean = freqs_clean >= 0
        
        # 크기 스펙트럼 계산
        magnitude_orig = np.abs(fft_orig[pos_mask_orig])
        magnitude_clean = np.abs(fft_clean[pos_mask_clean])
        
        # dB 스케일로 변환
        magnitude_orig_db = 20 * np.log10(magnitude_orig + 1e-10)
        magnitude_clean_db = 20 * np.log10(magnitude_clean + 1e-10)
        
        # 플롯 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 주파수 스펙트럼 비교
        ax1.plot(freqs_orig[pos_mask_orig], magnitude_orig_db, 'r-', alpha=0.7, label='원본 (노이즈 포함)')
        ax1.plot(freqs_clean[pos_mask_clean], magnitude_clean_db, 'b-', alpha=0.7, label='노이즈 제거됨')
        ax1.set_title('주파수 도메인 비교', fontsize=14, fontweight='bold')
        ax1.set_xlabel('주파수 (Hz)')
        ax1.set_ylabel('크기 (dB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, sr_orig/2)  # 나이퀴스트 주파수까지만
        
        # 파워 스펙트럼 밀도 (PSD) 비교
        f_orig, psd_orig = signal.welch(y_orig, sr_orig, nperseg=1024)
        f_clean, psd_clean = signal.welch(y_clean, sr_clean, nperseg=1024)
        
        ax2.semilogy(f_orig, psd_orig, 'r-', alpha=0.7, label='원본 (노이즈 포함)')
        ax2.semilogy(f_clean, psd_clean, 'b-', alpha=0.7, label='노이즈 제거됨')
        ax2.set_title('파워 스펙트럼 밀도 (PSD) 비교', fontsize=14, fontweight='bold')
        ax2.set_xlabel('주파수 (Hz)')
        ax2.set_ylabel('PSD (V²/Hz)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"주파수 도메인 비교 이미지 저장: {save_path}")
        
        plt.show()
    
    def analyze_noise_characteristics(self, original_path, denoised_path):
        """
        노이즈 특성 분석
        
        Args:
            original_path (str): 원본 오디오 파일 경로
            denoised_path (str): 노이즈 제거된 오디오 파일 경로
        """
        # 오디오 로드
        y_orig, sr_orig = self.load_audio(original_path)
        y_clean, sr_clean = self.load_audio(denoised_path)
        
        if y_orig is None or y_clean is None:
            print("오디오 파일을 로드할 수 없습니다.")
            return
        
        # 기본 통계
        print("=== 노이즈 제거 분석 결과 ===")
        print(f"원본 오디오:")
        print(f"  - RMS 에너지: {np.sqrt(np.mean(y_orig**2)):.6f}")
        print(f"  - 최대 진폭: {np.max(np.abs(y_orig)):.6f}")
        print(f"  - 동적 범위: {20*np.log10(np.max(np.abs(y_orig))/np.sqrt(np.mean(y_orig**2))):.2f} dB")
        
        print(f"\n노이즈 제거된 오디오:")
        print(f"  - RMS 에너지: {np.sqrt(np.mean(y_clean**2)):.6f}")
        print(f"  - 최대 진폭: {np.max(np.abs(y_clean)):.6f}")
        print(f"  - 동적 범위: {20*np.log10(np.max(np.abs(y_clean))/np.sqrt(np.mean(y_clean**2))):.2f} dB")
        
        # 신호 대 잡음비 추정
        if len(y_orig) == len(y_clean):
            noise_estimate = y_orig - y_clean
            signal_power = np.mean(y_clean**2)
            noise_power = np.mean(noise_estimate**2)
            snr_improvement = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            print(f"\n추정 SNR 개선: {snr_improvement:.2f} dB")
        
        # 스펙트럴 중심 주파수
        spectral_centroids_orig = librosa.feature.spectral_centroid(y=y_orig, sr=sr_orig)[0]
        spectral_centroids_clean = librosa.feature.spectral_centroid(y=y_clean, sr=sr_clean)[0]
        
        print(f"\n스펙트럴 중심 주파수:")
        print(f"  - 원본: {np.mean(spectral_centroids_orig):.2f} Hz")
        print(f"  - 노이즈 제거됨: {np.mean(spectral_centroids_clean):.2f} Hz")
    
    def create_comprehensive_analysis(self, original_path, denoised_path, output_dir="analysis_results"):
        """
        종합적인 주파수 분석 및 시각화
        
        Args:
            original_path (str): 원본 오디오 파일 경로
            denoised_path (str): 노이즈 제거된 오디오 파일 경로
            output_dir (str): 결과 저장 디렉토리
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== 종합적인 오디오 주파수 분석 시작 ===")
        
        # 1. 파형 비교
        print("1. 파형 비교 분석 중...")
        waveform_path = os.path.join(output_dir, "waveform_comparison.png")
        self.plot_waveform_comparison(original_path, denoised_path, waveform_path)
        
        # 2. 스펙트로그램 비교
        print("2. 스펙트로그램 비교 분석 중...")
        spectrogram_path = os.path.join(output_dir, "spectrogram_comparison.png")
        self.plot_spectrogram_comparison(original_path, denoised_path, spectrogram_path)
        
        # 3. 주파수 도메인 분석
        print("3. 주파수 도메인 분석 중...")
        frequency_path = os.path.join(output_dir, "frequency_domain_comparison.png")
        self.plot_frequency_domain_comparison(original_path, denoised_path, frequency_path)
        
        # 4. 노이즈 특성 분석
        print("4. 노이즈 특성 분석 중...")
        self.analyze_noise_characteristics(original_path, denoised_path)
        
        print(f"\n✅ 모든 분석 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")

def demo_frequency_visualization():
    """
    주파수 시각화 데모 함수
    """
    print("=== 오디오 주파수 시각화 데모 ===")
    
    # 시각화 객체 생성
    visualizer = AudioFrequencyVisualizer()
    
    # 데모용 오디오 파일 경로 (실제 파일로 변경하세요)
    original_file = "test.mp3"  # 원본 노이즈 파일
    denoised_file = "denoised_speech_hf.wav"  # 노이즈 제거된 파일
    
    # 파일 존재 확인
    if not os.path.exists(original_file):
        print(f"⚠️  원본 파일을 찾을 수 없습니다: {original_file}")
        print("실제 오디오 파일 경로로 변경하거나 파일을 생성하세요.")
        return
    
    if not os.path.exists(denoised_file):
        print(f"⚠️  노이즈 제거된 파일을 찾을 수 없습니다: {denoised_file}")
        print("먼저 노이즈 제거를 실행하세요.")
        
        # 노이즈 제거 실행
        print("노이즈 제거를 실행합니다...")
        try:
            from rnnaudio_huggingface import denoiser_model
            apply_denoising_huggingface(original_file, denoised_file, denoiser_model)
        except Exception as e:
            print(f"노이즈 제거 실패: {e}")
            return
    
    # 종합적인 분석 실행
    visualizer.create_comprehensive_analysis(original_file, denoised_file)

if __name__ == "__main__":
    demo_frequency_visualization()
