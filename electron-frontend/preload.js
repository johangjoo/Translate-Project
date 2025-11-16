const { contextBridge, ipcRenderer } = require('electron');

// 렌더러 프로세스에서 사용할 API 노출
contextBridge.exposeInMainWorld('electronAPI', {
  // 파일 선택
  selectFile: () => ipcRenderer.invoke('select-file'),
  
  // 저장 위치 선택
  selectSaveLocation: () => ipcRenderer.invoke('select-save-location'),
  
  // 오디오 변환 (outputPath 파라미터 제거)
  convertToWav: (inputPath) => ipcRenderer.invoke('convert-to-wav', inputPath),
  
  // API 서버에 파일 전송 (옵션 포함)
  sendToAPI: (filePath, endpoint, serverUrl, options) =>
    ipcRenderer.invoke('send-to-api', filePath, endpoint, serverUrl, options),
  
  // 서버 상태 확인
  checkServerStatus: (serverUrl) => ipcRenderer.invoke('check-server-status', serverUrl),
  
  // 텍스트 번역
  translateText: (text, sourceLang, targetLang, serverUrl) => ipcRenderer.invoke('translate-text', text, sourceLang, targetLang, serverUrl),
  
  // 임시 파일 정리
  cleanupTempFiles: (filePaths) => ipcRenderer.invoke('cleanup-temp-files', filePaths),
  
  // 이벤트 리스너
  onConversionProgress: (callback) => {
    ipcRenderer.on('conversion-progress', (event, progress) => callback(progress));
  },
  
  onUploadProgress: (callback) => {
    ipcRenderer.on('upload-progress', (event, progress) => callback(progress));
  },
  
  // 이벤트 리스너 제거
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
}
);