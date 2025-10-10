"""
오디오 주파수 시각화 테스트 스크립트
"""

import os
import numpy as np
import soundfile as sf
from audio_frequency_visualizer import AudioFrequencyVisualizer
from rnnaudio_huggingface import HuggingFaceDenoiser, apply_denoising_huggingface

def create_test_audio():
    """
    테스트용 노이즈가 포함된 오디오 생성
    """
    print("테스트용 오디오 파일 생성 중...")
    
    # 파라미터
    sr = 16000
    duration = 5  # 5초
    t = np.linspace(0, duration, sr * duration)
    
    # 음성 유사 신호 생성 (여러 주파수 성분)
    speech_signal = (
        # 기본 주파수 (200Hz) - 남성 음성 기본 주파수
        0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 0.1) +
        # 첫 번째 하모닉 (400Hz)
        0.3 * np.sin(2 * np.pi * 400 * t) * np.exp(-t * 0.15) +
        # 두 번째 하모닉 (600Hz)
        0.2 * np.sin(2 * np.pi * 600 * t) * np.exp(-t * 0.2) +
        # 고주파 성분 (1200Hz) - 자음 소리
        0.1 * np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 0.3) +
        # 포먼트 주파수 (800Hz)
        0.15 * np.sin(2 * np.pi * 800 * t) * np.exp(-t * 0.25)
    )
    
    # 시간에 따른 진폭 변조 (자연스러운 음성 패턴)
    envelope = np.exp(-0.5 * ((t - duration/2) / (duration/4))**2)  # 가우시안 엔벨로프
    speech_signal *= envelope
    
    # 다양한 종류의 노이즈 추가
    # 1. 백색 노이즈
    white_noise = np.random.randn(len(speech_signal)) * 0.2
    
    # 2. 핑크 노이즈 (1/f 노이즈)
    pink_noise = np.random.randn(len(speech_signal))
    # 간단한 핑크 노이즈 필터 적용
    for i in range(1, len(pink_noise)):
        pink_noise[i] = 0.99 * pink_noise[i-1] + 0.1 * pink_noise[i]
    pink_noise *= 0.15
    
    # 3. 60Hz 허밍 노이즈 (전원 노이즈)
    hum_noise = 0.1 * np.sin(2 * np.pi * 60 * t)
    
    # 4. 고주파 노이즈 (에어컨, 팬 소음 등)
    high_freq_noise = 0.05 * np.sin(2 * np.pi * 3000 * t) * np.random.randn(len(t)) * 0.5
    
    # 모든 노이즈 결합
    total_noise = white_noise + pink_noise + hum_noise + high_freq_noise
    
    # 신호와 노이즈 결합
    noisy_signal = speech_signal + total_noise
    
    # 정규화 (클리핑 방지)
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 0:
        noisy_signal = noisy_signal / max_val * 0.8
    
    # 파일 저장
    test_file = "test_noisy_audio.wav"
    sf.write(test_file, noisy_signal, sr)
    print(f"✅ 테스트 오디오 생성 완료: {test_file}")
    
    return test_file

def test_frequency_visualization():
    """
    주파수 시각화 테스트 실행
    """
    print("=== 오디오 주파수 시각화 테스트 ===\n")
    
    # 1. 테스트 오디오 생성
    original_file = create_test_audio()
    denoised_file = "test_denoised_audio.wav"
    
    # 2. 노이즈 제거 실행
    print("\n노이즈 제거 실행 중...")
    try:
        # Hugging Face 모델 초기화
        denoiser = HuggingFaceDenoiser(model_type="spectral_mask", use_gpu=False)
        
        # 노이즈 제거 적용
        result_path = apply_denoising_huggingface(original_file, denoised_file, denoiser)
        print(f"✅ 노이즈 제거 완료: {result_path}")
        
    except Exception as e:
        print(f"⚠️  노이즈 제거 중 오류: {e}")
        print("기본 처리 방법을 사용합니다...")
        return
    
    # 3. 주파수 시각화 실행
    print("\n주파수 분석 및 시각화 시작...")
    
    # 시각화 객체 생성
    visualizer = AudioFrequencyVisualizer(sr=16000)
    
    try:
        # 종합적인 분석 실행
        output_dir = "frequency_analysis_results"
        visualizer.create_comprehensive_analysis(original_file, denoised_file, output_dir)
        
        print(f"\n🎉 모든 테스트 완료!")
        print(f"결과 파일들:")
        print(f"  - 원본 오디오: {original_file}")
        print(f"  - 노이즈 제거된 오디오: {denoised_file}")
        print(f"  - 분석 결과: {output_dir}/ 폴더")
        
        # 개별 시각화 함수들도 테스트
        print("\n추가 시각화 테스트...")
        
        # 파형 비교만 따로 실행
        print("- 파형 비교 시각화...")
        visualizer.plot_waveform_comparison(original_file, denoised_file)
        
        # 스펙트로그램 비교만 따로 실행  
        print("- 스펙트로그램 비교 시각화...")
        visualizer.plot_spectrogram_comparison(original_file, denoised_file)
        
        # 주파수 도메인 분석만 따로 실행
        print("- 주파수 도메인 분석...")
        visualizer.plot_frequency_domain_comparison(original_file, denoised_file)
        
    except Exception as e:
        print(f"❌ 시각화 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_with_real_audio():
    """
    실제 오디오 파일로 테스트 (파일이 있는 경우)
    """
    print("=== 실제 오디오 파일 테스트 ===")
    
    # 실제 파일 경로들 (존재하는 경우에만 실행)
    real_files = ["test.mp3", "input/audio.wav", "noisy_speech.wav"]
    
    for file_path in real_files:
        if os.path.exists(file_path):
            print(f"\n실제 파일 발견: {file_path}")
            
            # 노이즈 제거된 파일 경로
            base_name = os.path.splitext(file_path)[0]
            denoised_file = f"{base_name}_denoised.wav"
            
            # 노이즈 제거 실행
            try:
                denoiser = HuggingFaceDenoiser(model_type="spectral_mask", use_gpu=False)
                apply_denoising_huggingface(file_path, denoised_file, denoiser)
                
                # 시각화 실행
                visualizer = AudioFrequencyVisualizer()
                output_dir = f"analysis_{base_name}"
                visualizer.create_comprehensive_analysis(file_path, denoised_file, output_dir)
                
                print(f"✅ {file_path} 분석 완료!")
                
            except Exception as e:
                print(f"⚠️  {file_path} 처리 중 오류: {e}")
            
            break  # 첫 번째 파일만 처리
    else:
        print("실제 오디오 파일을 찾을 수 없습니다.")
        print("다음 중 하나의 파일을 준비하세요:")
        for file_path in real_files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    # 기본 테스트 실행
    test_frequency_visualization()
    
    # 실제 파일이 있다면 추가 테스트
    print("\n" + "="*50)
    test_with_real_audio()
    
    print("\n🎯 테스트 완료!")
    print("\n사용법:")
    print("1. audio_frequency_visualizer.py를 import하여 사용")
    print("2. AudioFrequencyVisualizer 클래스의 메서드들을 호출")
    print("3. create_comprehensive_analysis()로 전체 분석 실행")
    print("4. 개별 plot_* 메서드들로 특정 분석만 실행")
