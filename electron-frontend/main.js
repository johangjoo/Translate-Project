const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const ffmpeg = require('fluent-ffmpeg');
const axios = require('axios');
const FormData = require('form-data');
const os = require('os');

// 개발 모드 확인
const isDev = process.argv.includes('--dev');

// 메인 윈도우 참조
let mainWindow;

// ✅ FFmpeg 바이너리 경로 자동 감지
function getFfmpegPath() {
  const platform = process.platform;
  let ffmpegName = 'ffmpeg';
  
  if (platform === 'win32') {
    ffmpegName = 'ffmpeg.exe';
  }
  
  console.log(`🔍 FFmpeg 경로 탐색 중... (플랫폼: ${platform})`);
  
  // 1️⃣ 프로덕션 환경 (배포된 앱) - 번들된 FFmpeg 사용
  if (app.isPackaged) {
    const resourcesPath = process.resourcesPath;
    let platformDir;
    
    if (platform === 'win32') {
      platformDir = 'win';
    } else if (platform === 'darwin') {
      platformDir = 'mac';
    } else {
      platformDir = 'linux';
    }
    
    const ffmpegPath = path.join(resourcesPath, 'ffmpeg', platformDir, ffmpegName);
    
    if (fs.existsSync(ffmpegPath)) {
      console.log('✅ 번들된 FFmpeg 사용:', ffmpegPath);
      return ffmpegPath;
    } else {
      console.error('❌ 번들된 FFmpeg를 찾을 수 없습니다:', ffmpegPath);
    }
  }
  
  // 2️⃣ 개발 환경 - npm 패키지 FFmpeg 사용
  try {
    const ffmpegInstaller = require('@ffmpeg-installer/ffmpeg');
    if (fs.existsSync(ffmpegInstaller.path)) {
      console.log('✅ npm FFmpeg 사용 (개발 모드):', ffmpegInstaller.path);
      return ffmpegInstaller.path;
    }
  } catch (e) {
    console.log('⚠️ npm FFmpeg를 찾을 수 없습니다.');
  }
  
  // 3️⃣ 시스템 설치된 FFmpeg 사용 (개발자가 직접 설치한 경우)
  const systemPaths = [
    'C:\\ffmpeg-2025-09-28-git-0fdb5829e3-full_build\\bin\\ffmpeg.exe',
    'C:\\ffmpeg\\bin\\ffmpeg.exe',
    '/usr/local/bin/ffmpeg',
    '/usr/bin/ffmpeg'
  ];
  
  for (const systemPath of systemPaths) {
    if (fs.existsSync(systemPath)) {
      console.log('✅ 시스템 FFmpeg 사용:', systemPath);
      return systemPath;
    }
  }
  
  // 4️⃣ 시스템 PATH에서 FFmpeg 찾기
  console.log('ℹ️ 시스템 PATH의 FFmpeg 사용 시도');
  return null; // fluent-ffmpeg가 자동으로 PATH에서 찾음
}

function createWindow() {
  // 메인 윈도우 생성
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    show: false,
    titleBarStyle: 'default'
  });

  // HTML 파일 로드
  mainWindow.loadFile('renderer/index.html');

  // 개발 모드에서 DevTools 열기
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // 윈도우가 준비되면 보여주기
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // 윈도우가 닫힐 때
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// 앱이 준비되면 윈도우 생성
app.whenReady().then(() => {
  console.log('🚀 앱 시작');
  console.log(`   패키징 상태: ${app.isPackaged ? '배포 모드' : '개발 모드'}`);
  console.log(`   앱 경로: ${app.getAppPath()}`);
  if (app.isPackaged) {
    console.log(`   리소스 경로: ${process.resourcesPath}`);
  }
  
  createWindow();
});

// 모든 윈도우가 닫혔을 때
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// 앱이 활성화될 때 (macOS)
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ===== IPC 핸들러 =====

// 파일 선택 다이얼로그
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Audio/Video Files', extensions: ['mp3', 'wav', 'mp4', 'avi', 'mov', 'mkv', 'flv', 'webm', 'm4a', 'aac', 'ogg', 'flac'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return result;
});

// 저장 위치 선택 다이얼로그
ipcMain.handle('select-save-location', async () => {
  const result = await dialog.showSaveDialog(mainWindow, {
    filters: [
      { name: 'WAV Files', extensions: ['wav'] }
    ],
    defaultPath: 'converted_audio.wav'
  });
  
  return result;
});

// ✅ FFmpeg를 사용한 오디오 변환 (완전 개선판)
ipcMain.handle('convert-to-wav', async (event, inputPath) => {
  return new Promise((resolve, reject) => {
    try {
      // ✅ Main process에서 임시 파일 경로 생성
      const timestamp = Date.now();
      const tempDir = os.tmpdir();
      const outputPath = path.join(tempDir, `audio_${timestamp}.wav`);
      
      console.log(`\n${'='.repeat(60)}`);
      console.log(`🎵 오디오 변환 시작`);
      console.log(`${'='.repeat(60)}`);
      console.log(`   입력: ${inputPath}`);
      console.log(`   출력: ${outputPath}`);
      
      // ✅ FFmpeg 경로 설정
      const ffmpegPath = getFfmpegPath();
      if (ffmpegPath) {
        console.log(`   FFmpeg: ${ffmpegPath}`);
        ffmpeg.setFfmpegPath(ffmpegPath);
      } else {
        console.log(`   FFmpeg: 시스템 PATH 사용`);
      }
      
      // ✅ 입력 파일 존재 확인
      if (!fs.existsSync(inputPath)) {
        const error = `입력 파일을 찾을 수 없습니다: ${inputPath}`;
        console.error('❌', error);
        reject({ success: false, error });
        return;
      }
      
      console.log(`${'='.repeat(60)}\n`);
      
      // 진행률 업데이트를 위한 함수
      const updateProgress = (progress) => {
        event.sender.send('conversion-progress', progress);
      };

      ffmpeg(inputPath)
        .toFormat('wav')
        .audioCodec('pcm_s16le')
        .audioChannels(1)
        .audioFrequency(16000)
        .on('start', (commandLine) => {
          console.log('▶️  FFmpeg 명령어:', commandLine);
          updateProgress({ status: 'started', percent: 0 });
        })
        .on('progress', (progress) => {
          const percent = progress.percent || 0;
          if (percent > 0) {
            process.stdout.write(`\r⏳ 변환 진행률: ${Math.round(percent)}%`);
          }
          updateProgress({ 
            status: 'processing', 
            percent: percent,
            timemark: progress.timemark
          });
        })
        .on('end', () => {
          console.log('\n✅ 변환 완료:', outputPath);
          
          // ✅ 출력 파일 생성 확인
          if (fs.existsSync(outputPath)) {
            const stats = fs.statSync(outputPath);
            console.log(`   파일 크기: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
            console.log(`${'='.repeat(60)}\n`);
            
            updateProgress({ status: 'completed', percent: 100 });
            resolve({ success: true, outputPath });
          } else {
            const error = '변환은 완료되었으나 출력 파일이 생성되지 않았습니다.';
            console.error('❌', error);
            updateProgress({ status: 'error', error });
            reject({ success: false, error });
          }
        })
        .on('error', (err, stdout, stderr) => {
          console.error('\n❌ 변환 오류:', err.message);
          if (stdout) console.error('   stdout:', stdout);
          if (stderr) console.error('   stderr:', stderr);
          console.log(`${'='.repeat(60)}\n`);
          
          updateProgress({ status: 'error', error: err.message });
          reject({ success: false, error: err.message });
        })
        .save(outputPath);
        
    } catch (error) {
      console.error('❌ 변환 초기화 오류:', error);
      reject({ success: false, error: error.message });
    }
  });
});

// ✅ FastAPI 서버에 오디오 파일 전송
ipcMain.handle('send-to-api', async (event, filePath, apiEndpoint, serverUrl = 'http://127.0.0.1:8000', options = {}) => {

  try {
    // 파일 존재 확인
    if (!fs.existsSync(filePath)) {
      throw new Error('파일을 찾을 수 없습니다.');
    }

    // FormData 생성
    const formData = new FormData();
    formData.append('audio_file', fs.createReadStream(filePath));
    
    // ✅ FastAPI 엔드포인트에 맞는 파라미터 추가
    if (apiEndpoint === 'audio/process') {
      formData.append('enable_denoise', 'false');
      formData.append('enable_transcription', 'true');
      formData.append('enable_diarization', 'true');
      formData.append('save_outputs', 'false');

      // 최대 화자 수 전달 (선택 사항)
      if (options && typeof options.maxSpeakers === 'number') {
        formData.append('max_speakers', String(options.maxSpeakers));
      }
    }

    // ✅ API 요청
    console.log(`🌐 API 요청: ${serverUrl}/api/v1/${apiEndpoint}`);
    const response = await axios.post(`${serverUrl}/api/v1/${apiEndpoint}`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 300000, // 5분 타임아웃
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        event.sender.send('upload-progress', { percent: percentCompleted });
        console.log(`📤 업로드 진행률: ${percentCompleted}%`);
      }
    });

    console.log('✅ API 응답 성공');
    return { success: true, data: response.data };
  } catch (error) {
    console.error('❌ API 요청 오류:', error.message);
    if (error.response) {
      console.error('   응답 상태:', error.response.status);
      console.error('   응답 데이터:', error.response.data);
    }
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message 
    };
  }
});

// ✅ 서버 상태 확인
ipcMain.handle('check-server-status', async (event, serverUrl = 'http://127.0.0.1:8000') => {
  try {
    console.log(`🔍 서버 상태 확인: ${serverUrl}/api/v1/health`);
    const response = await axios.get(`${serverUrl}/api/v1/health`, { timeout: 5000 });
    console.log('✅ 서버 연결 성공:', response.data);
    return { success: true, data: response.data };
  } catch (error) {
    console.error('❌ 서버 연결 실패:', error.message);
    return { success: false, error: error.message };
  }
});

// ✅ 텍스트 번역
ipcMain.handle('translate-text', async (event, text, sourceLang, targetLang, serverUrl = 'http://127.0.0.1:8000') => {
  try {
    console.log(`🌐 텍스트 번역 요청: ${sourceLang} → ${targetLang}`);
    
    // Form 데이터로 전송
    const URLSearchParams = require('url').URLSearchParams;
    const params = new URLSearchParams({
      text: text,
      source_lang: sourceLang,
      target_lang: targetLang
    });
    
    const response = await axios.post(`${serverUrl}/api/v1/translate-text`,
      params.toString(),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        timeout: 60000
      }
    );

    console.log('✅ 번역 완료');
    return { success: true, data: response.data };
  } catch (error) {
    console.error('❌ 텍스트 번역 오류:', error.message);
    if (error.response) {
      console.error('   응답 상태:', error.response.status);
      console.error('   응답 데이터:', error.response.data);
    }
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message 
    };
  }
});

// 임시 파일 정리
ipcMain.handle('cleanup-temp-files', async (event, filePaths) => {
  try {
    for (const filePath of filePaths) {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        console.log('🗑️ 임시 파일 삭제:', filePath);
      }
    }
    return { success: true };
  } catch (error) {
    console.error('❌ 임시 파일 정리 오류:', error);
    return { success: false, error: error.message };
  }
});

// 앱 종료 시 정리 작업
app.on('before-quit', () => {
  console.log('👋 앱 종료 중...');
});