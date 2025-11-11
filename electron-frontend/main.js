const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const ffmpeg = require('fluent-ffmpeg');
const axios = require('axios');
const FormData = require('form-data');

// 개발 모드 확인
const isDev = process.argv.includes('--dev');

// 메인 윈도우 참조
let mainWindow;

// FFmpeg 바이너리 경로 설정 (필요시)
// ffmpeg.setFfmpegPath('path/to/ffmpeg');
// ffmpeg.setFfprobePath('path/to/ffprobe');

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
app.whenReady().then(createWindow);

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

// FFmpeg를 사용한 오디오 변환
ipcMain.handle('convert-to-wav', async (event, inputPath, outputPath) => {
  return new Promise((resolve, reject) => {
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
        console.log('FFmpeg 명령어:', commandLine);
        updateProgress({ status: 'started', percent: 0 });
      })
      .on('progress', (progress) => {
        console.log('변환 진행률:', progress.percent + '%');
        updateProgress({ 
          status: 'processing', 
          percent: progress.percent || 0,
          timemark: progress.timemark
        });
      })
      .on('end', () => {
        console.log('변환 완료');
        updateProgress({ status: 'completed', percent: 100 });
        resolve({ success: true, outputPath });
      })
      .on('error', (err) => {
        console.error('변환 오류:', err);
        updateProgress({ status: 'error', error: err.message });
        reject({ success: false, error: err.message });
      })
      .save(outputPath);
  });
});

// FastAPI 서버에 오디오 파일 전송
ipcMain.handle('send-to-api', async (event, filePath, apiEndpoint, serverUrl = 'http://127.0.0.1:8000') => {
  try {
    // 파일 존재 확인
    if (!fs.existsSync(filePath)) {
      throw new Error('파일을 찾을 수 없습니다.');
    }

    // FormData 생성
    const formData = new FormData();
    // ✅ 'audio_file'로 필드명 통일
    formData.append('audio_file', fs.createReadStream(filePath));

    // API 요청
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

// 서버 상태 확인
ipcMain.handle('check-server-status', async (event, serverUrl = 'http://http://127.0.0.1:8000') => {
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

// 텍스트 번역 (오디오 없이)
ipcMain.handle('translate-text', async (event, text, sourceLang, targetLang, serverUrl = 'http://127.0.0.1:8000') => {
  try {
    console.log(`🌐 텍스트 번역 요청: ${sourceLang} → ${targetLang}`);
    
    // ✅ Form 데이터로 전송 (FastAPI Form과 일치)
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