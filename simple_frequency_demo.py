"""
간단한 주파수 시각화 데모
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy import signal

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_simple_test_audio():
    """간단한 테스트 오디오 생성"""
    sr = 16000
    duration = 2  # 2초
    t = np.linspace(0, duration, sr * duration)
    
    # 간단한 신호 생성
    clean_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz 사인파 (라 음)
    noise = np.random.randn(len(clean_signal)) * 0.3  # 노이즈
    noisy_signal = clean_signal + noise
    
    # 파일 저장
    sf.write("simple_clean.wav", clean_signal, sr)
    sf.write("simple_noisy.wav", noisy_signal, sr)
    
    return "simple_noisy.wav", "simple_clean.wav"

def plot_simple_frequency_comparison():
    """간단한 주파수 비교 시각화"""
    # 테스트 오디오 생성
    noisy_file, clean_file = create_simple_test_audio()
    
    # 오디오 로드
    y_noisy, sr = librosa.load(noisy_file, sr=16000)
    y_clean, sr = librosa.load(clean_file, sr=16000)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 파형 비교
    t = np.linspace(0, len(y_noisy)/sr, len(y_noisy))
    axes[0, 0].plot(t, y_noisy, 'r-', alpha=0.7, label='노이즈 포함')
    axes[0, 0].plot(t, y_clean, 'b-', alpha=0.7, label='깨끗한 신호')
    axes[0, 0].set_title('파형 비교')
    axes[0, 0].set_xlabel('시간 (초)')
    axes[0, 0].set_ylabel('진폭')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 주파수 스펙트럼
    fft_noisy = np.fft.fft(y_noisy)
    fft_clean = np.fft.fft(y_clean)
    freqs = np.fft.fftfreq(len(fft_noisy), 1/sr)
    
    # 양의 주파수만
    pos_mask = freqs >= 0
    axes[0, 1].plot(freqs[pos_mask], np.abs(fft_noisy[pos_mask]), 'r-', alpha=0.7, label='노이즈 포함')
    axes[0, 1].plot(freqs[pos_mask], np.abs(fft_clean[pos_mask]), 'b-', alpha=0.7, label='깨끗한 신호')
    axes[0, 1].set_title('주파수 스펙트럼')
    axes[0, 1].set_xlabel('주파수 (Hz)')
    axes[0, 1].set_ylabel('크기')
    axes[0, 1].set_xlim(0, 2000)  # 0-2kHz만 표시
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 스펙트로그램 (노이즈 포함)
    D_noisy = librosa.stft(y_noisy)
    DB_noisy = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)
    img1 = librosa.display.specshow(DB_noisy, sr=sr, x_axis='time', y_axis='hz', ax=axes[1, 0])
    axes[1, 0].set_title('스펙트로그램 (노이즈 포함)')
    plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')
    
    # 4. 스펙트로그램 (깨끗한 신호)
    D_clean = librosa.stft(y_clean)
    DB_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
    img2 = librosa.display.specshow(DB_clean, sr=sr, x_axis='time', y_axis='hz', ax=axes[1, 1])
    axes[1, 1].set_title('스펙트로그램 (깨끗한 신호)')
    plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('simple_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 간단한 주파수 분석 완료!")
    print("결과 이미지: simple_frequency_analysis.png")

def analyze_audio_characteristics(audio_file):
    """오디오 특성 분석"""
    if not os.path.exists(audio_file):
        print(f"파일을 찾을 수 없습니다: {audio_file}")
        return
    
    # 오디오 로드
    y, sr = librosa.load(audio_file, sr=16000)
    
    print(f"=== {audio_file} 분석 결과 ===")
    print(f"샘플링 레이트: {sr} Hz")
    print(f"길이: {len(y)/sr:.2f} 초")
    print(f"RMS 에너지: {np.sqrt(np.mean(y**2)):.6f}")
    print(f"최대 진폭: {np.max(np.abs(y)):.6f}")
    
    # 스펙트럴 특성
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    print(f"스펙트럴 중심 주파수: {np.mean(spectral_centroids):.2f} Hz")
    print(f"스펙트럴 롤오프: {np.mean(spectral_rolloff):.2f} Hz")
    print(f"영점 교차율: {np.mean(zero_crossing_rate):.4f}")

if __name__ == "__main__":
    import os
    
    print("=== 간단한 주파수 시각화 데모 ===")
    
    # 간단한 시각화 실행
    plot_simple_frequency_comparison()
    
    # 기존 파일들이 있다면 분석
    test_files = ["test.mp3", "denoised_speech_hf.wav", "simple_noisy.wav"]
    
    for file in test_files:
        if os.path.exists(file):
            print(f"\n{file} 분석:")
            analyze_audio_characteristics(file)
    
    print("\n🎉 데모 완료!")
