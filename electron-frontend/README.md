# Audio Translation Frontend

Electron 기반의 음성 번역 프론트엔드 애플리케이션입니다. FFmpeg를 사용하여 다양한 오디오/비디오 파일을 WAV 형식으로 변환하고, FastAPI 서버에 전송하여 음성 인식 및 번역을 수행합니다.

## 주요 기능

- 🎵 **다양한 형식 지원**: MP3, WAV, MP4, AVI, MOV, MKV 등
- 🔄 **FFmpeg 통합**: 자동 오디오 변환 (WAV 16kHz mono)
- 🎤 **음성 인식**: Whisper 기반 STT (99개 언어 지원)
- 🌐 **번역**: Qwen3-8b 기반 번역 (한국어 ↔ 일본어)
- 📊 **실시간 진행률**: 변환, 업로드, 처리 진행률 표시
- 💾 **결과 저장**: 음성 인식 및 번역 결과 저장
- 📝 **텍스트 번역**: 오디오 없이 텍스트만 번역

## 시스템 요구사항

- **Node.js**: 16.0 이상
- **FFmpeg**: 시스템에 설치되어 있어야 함
- **FastAPI 서버**: 백엔드 서버가 실행 중이어야 함

## 설치 방법

### 1. 의존성 설치

```bash
cd electron-frontend
npm install
```

### 2. FFmpeg 설치

#### Windows
```bash
# Chocolatey 사용
choco install ffmpeg

# 또는 수동 설치
# https://ffmpeg.org/download.html 에서 다운로드
```

#### macOS
```bash
# Homebrew 사용
brew install ffmpeg
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

### 3. 백엔드 서버 실행

먼저 FastAPI 서버를 실행해야 합니다:

```bash
# 프로젝트 루트에서
cd ..
python run_server.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 실행 방법

### 개발 모드
```bash
npm run dev
```

### 프로덕션 모드
```bash
npm start
```

### 빌드
```bash
# 패키징만
npm run pack

# 배포용 빌드
npm run build
```

## 사용 방법

### 1. 서버 연결 확인
- 앱 상단의 서버 URL이 올바른지 확인
- "연결 확인" 버튼으로 서버 상태 체크

### 2. 파일 선택
- "파일 선택" 버튼 클릭 또는 드래그 앤 드롭
- 지원 형식: MP3, WAV, MP4, AVI, MOV, MKV, FLV, WEBM, M4A, AAC, OGG, FLAC

### 3. 처리 모드 선택
- **음성 인식만**: 오디오를 텍스트로만 변환
- **음성 인식 + 번역**: 텍스트 변환 후 번역까지 수행

### 4. 처리 시작
- "처리 시작" 버튼 클릭
- 진행률을 실시간으로 확인

### 5. 결과 확인
- 음성 인식 결과 및 번역 결과 확인
- 클립보드 복사 또는 파일 저장

## 프로젝트 구조

```
electron-frontend/
├── main.js              # 메인 프로세스
├── preload.js           # 프리로드 스크립트
├── package.json         # 패키지 설정
├── renderer/            # 렌더러 프로세스
│   ├── index.html       # 메인 HTML
│   ├── styles.css       # 스타일시트
│   └── script.js        # 클라이언트 로직
└── assets/              # 리소스 파일
    └── icon.png         # 앱 아이콘
```

## API 엔드포인트

애플리케이션은 다음 FastAPI 엔드포인트를 사용합니다:

- `GET /api/v1/health` - 서버 상태 확인
- `POST /api/v1/transcribe` - 음성 인식만
- `POST /api/v1/audio-to-translation` - 음성 인식 + 번역
- `POST /api/v1/translate-text` - 텍스트 번역

## 설정

### FFmpeg 경로 설정
시스템에 FFmpeg가 설치되어 있지만 PATH에 없는 경우, `main.js`에서 경로를 직접 설정할 수 있습니다:

```javascript
const ffmpeg = require('fluent-ffmpeg');
ffmpeg.setFfmpegPath('/path/to/ffmpeg');
ffmpeg.setFfprobePath('/path/to/ffprobe');
```

### 서버 URL 변경
기본 서버 URL은 `http://localhost:8000`입니다. 다른 서버를 사용하려면 앱에서 직접 변경하거나 `renderer/script.js`의 기본값을 수정하세요.

## 문제 해결

### 1. FFmpeg 오류
```
Error: ffmpeg exited with code 1
```
- FFmpeg가 올바르게 설치되었는지 확인
- 터미널에서 `ffmpeg -version` 명령어 실행하여 확인

### 2. 서버 연결 오류
```
서버 연결 실패: Network Error
```
- FastAPI 서버가 실행 중인지 확인
- 방화벽 설정 확인
- 서버 URL이 올바른지 확인

### 3. 파일 변환 오류
```
변환 실패: Invalid input format
```
- 지원되는 파일 형식인지 확인
- 파일이 손상되지 않았는지 확인

### 4. 메모리 부족
- 큰 파일의 경우 시스템 메모리가 부족할 수 있음
- 파일 크기를 줄이거나 시스템 메모리를 늘려보세요

## 개발

### 개발 환경 설정
```bash
# 의존성 설치
npm install

# 개발 모드 실행
npm run dev

# 코드 변경 시 자동 재시작을 위해 nodemon 사용 (선택사항)
npm install -g nodemon
nodemon --exec "npm run dev"
```

### 빌드 설정
`package.json`의 `build` 섹션에서 빌드 설정을 변경할 수 있습니다:

```json
{
  "build": {
    "appId": "com.example.audio-translation",
    "productName": "Audio Translation",
    "directories": {
      "output": "dist"
    },
    "win": {
      "target": "nsis"
    },
    "mac": {
      "target": "dmg"
    },
    "linux": {
      "target": "AppImage"
    }
  }
}
```

## 라이선스

MIT License

## 기여

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 지원

문제가 있거나 기능 요청이 있으면 GitHub Issues를 통해 알려주세요.
