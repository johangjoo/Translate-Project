import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import os

# --- 1. RNN 모델 구조 (이전 응답의 PyTorch 코드 재사용) ---
# 실제 사용 시에는 이 모델을 훈련된 가중치로 로드해야 합니다.
class RNNDenoiser(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, output_features, dropout=0.2):
        super(RNNDenoiser, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True # 양방향 RNN 사용 (hidden_size * 2)
        )
        # 양방향 RNN이므로 출력 크기가 2배가 됩니다.
        output_size = hidden_size * 2 if self.rnn.bidirectional else hidden_size
        self.fc = nn.Linear(output_size, output_features)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        rnn_out, _ = self.rnn(x) 
        mask = self.activation(self.fc(rnn_out)) 
        return mask

# --- 2. 하이퍼파라미터 및 설정 ---
SR = 16000          # 샘플링 레이트 (Hz)
N_FFT = 512         # FFT 윈도우 크기 (librosa 기본값: 2048. 음성에서는 512 권장)
HOP_LENGTH = 128    # 홉 길이 (프레임 간의 샘플 수)
# STFT 결과의 주파수 빈(Frequency bins) 수
INPUT_FEATURES = N_FFT // 2 + 1 
HIDDEN_SIZE = 256
NUM_LAYERS = 2
OUTPUT_FEATURES = INPUT_FEATURES

# --- 3. 훈련된 모델 인스턴스화 (예시를 위해 가짜 모델 생성) ---
# 실제 사용 시에는 torch.load() 등으로 훈련된 가중치를 로드해야 함
denoiser_model = RNNDenoiser(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_FEATURES)
denoiser_model.eval() # 추론 모드 설정

# --- 4. 음성 잡음 제거 함수 ---
def apply_denoising_rnn(noisy_audio_path, output_audio_path, model, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    
    # 1. Librosa를 이용한 오디오 로드 및 STFT
    # y: 오디오 파형 (time-domain signal), sr: 샘플링 레이트
    y_noisy, sr = librosa.load(noisy_audio_path, sr=sr, mono=True)
    
    # STFT: 복소수 스펙트로그램 (Complex Spectrogram) 획득
    # D_noisy 형태: (n_fft/2 + 1, n_frames)
    D_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length)
    
    # 크기(Magnitude)와 위상(Phase) 분리
    S_mag, S_phase = librosa.magphase(D_noisy)
    
    # 모델 입력 형태: (batch_size, time_steps, features)
    # RNN은 시간 축을 시퀀스로 처리하므로, (n_frames, n_freq_bins) 형태로 변환
    input_tensor = torch.from_numpy(S_mag.T).float().unsqueeze(0) # unsqueeze(0)로 batch_size 차원 추가

    # 2. RNN 모델을 이용한 마스크 예측
    with torch.no_grad(): # 추론 시 기울기 계산 비활성화
        predicted_mask = model(input_tensor).squeeze(0) # batch 차원 제거
        # predicted_mask 형태: (n_frames, n_freq_bins)
    
    # 마스크를 다시 (n_freq_bins, n_frames) 형태로 변환하여 스펙트로그램 크기와 일치
    mask_np = predicted_mask.numpy().T
    
    # 3. 마스크 적용 및 깨끗한 스펙트로그램 추정
    # Hadamard 곱 (요소별 곱셈): 잡음 제거된 크기 = noisy_mag * mask
    S_clean_mag_est = S_mag * mask_np 

    # 4. Librosa의 iSTFT를 이용한 음성 파형 복원
    # 잡음 제거된 크기와 원래 잡음 신호의 위상(Phase)을 결합
    D_clean_est = S_clean_mag_est * S_phase
    
    # iSTFT: 깨끗한 파형 (time-domain signal) 복원
    y_clean_est = librosa.istft(D_clean_est, hop_length=hop_length, length=len(y_noisy))
    
    # 5. 복원된 오디오 파일 저장
    sf.write(output_audio_path, y_clean_est, sr)
    
    print(f"잡음 제거 완료. 결과 파일: {output_audio_path}")
    return y_clean_est

# --- 5. 실행 예시 (가상 파일 사용) ---

# 더미 오디오 파일 생성 (실제 파일 경로로 대체해야 합니다)
dummy_noisy_path = "noisy_speech.wav"
dummy_output_path = "denoised_speech.wav"

if not os.path.exists(dummy_noisy_path):
    # 1초, 16kHz의 랜덤 노이즈 신호 생성
    dummy_signal = np.random.randn(SR * 1) * 0.5 
    sf.write(dummy_noisy_path, dummy_signal, SR)
    print(f"더미 노이즈 파일 생성: {dummy_noisy_path}")

# 잡음 제거 함수 실행
# y_denoised = apply_denoising_rnn(dummy_noisy_path, dummy_output_path, denoiser_model)
# print(f"복원된 음성 파형 길이: {len(y_denoised)}")

# 참고: 이 코드는 모델이 훈련되지 않았으므로 (가중치가 랜덤) 실제 잡음 제거 성능은 없습니다.
# 실제 사용을 위해서는 충분한 데이터셋으로 RNNDenoiser 모델을 훈련해야 합니다.
#
#  새로운 Hugging Face  버전 사용 가능!
# 더 나은 성능을 위해 rnnaudio_huggingface.py 파일을 사용하세요.
# 이 파일은 사전 훈련된 SpeechBrain 모델을 사용하여 실제 노이즈 제거 성능을 제공합니다.
# ----------------------------------------------------