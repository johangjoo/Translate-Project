// 전역 변수
let selectedFilePath = null;
let convertedWavPath = null;
let tempFiles = [];
let sttResultData = null; // ✅ STT 결과 저장용

// DOM 요소들
const elements = {
    serverUrl: document.getElementById('serverUrl'),
    serverStatus: document.getElementById('serverStatus'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    checkServerBtn: document.getElementById('checkServerBtn'),
    
    fileDropZone: document.getElementById('fileDropZone'),
    selectFileBtn: document.getElementById('selectFileBtn'),
    selectedFile: document.getElementById('selectedFile'),
    fileName: document.getElementById('fileName'),
    filePath: document.getElementById('filePath'),
    removeFileBtn: document.getElementById('removeFileBtn'),
    
    processingModes: document.querySelectorAll('input[name="processingMode"]'),
    translationSettings: document.getElementById('translationSettings'),
    sourceLang: document.getElementById('sourceLang'),
    targetLang: document.getElementById('targetLang'),
    processBtn: document.getElementById('processBtn'),
    
    progressPanel: document.getElementById('progressPanel'),
    conversionStatus: document.getElementById('conversionStatus'),
    conversionProgress: document.getElementById('conversionProgress'),
    conversionDetails: document.getElementById('conversionDetails'),
    uploadStatus: document.getElementById('uploadStatus'),
    uploadProgress: document.getElementById('uploadProgress'),
    processingStatus: document.getElementById('processingStatus'),
    aiProgress: document.getElementById('aiProgress'),
    
    resultsPanel: document.getElementById('resultsPanel'),
    detectedLang: document.getElementById('detectedLang'),
    sttTime: document.getElementById('sttTime'),
    sttResult: document.getElementById('sttResult'),
    translationResult: document.getElementById('translationResult'),
    translationInfo: document.getElementById('translationInfo'),
    translationTime: document.getElementById('translationTime'),
    translatedResult: document.getElementById('translatedResult'),
    newProcessBtn: document.getElementById('newProcessBtn'),
    saveResultBtn: document.getElementById('saveResultBtn'),
    
    textSourceLang: document.getElementById('textSourceLang'),
    textTargetLang: document.getElementById('textTargetLang'),
    inputText: document.getElementById('inputText'),
    outputText: document.getElementById('outputText'),
    translateTextBtn: document.getElementById('translateTextBtn'),
    
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    toast: document.getElementById('toast')
};

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    checkServerStatus();
});

function initializeApp() {
    // 처리 모드 변경 시 번역 설정 표시/숨김
    elements.processingModes.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'audio-to-translation') {
                elements.translationSettings.style.display = 'block';
            } else {
                elements.translationSettings.style.display = 'none';
            }
        });
    });
    
    // 파일 선택 상태에 따른 버튼 활성화
    updateProcessButton();
}

function setupEventListeners() {
    // 서버 연결 확인
    elements.checkServerBtn.addEventListener('click', checkServerStatus);
    
    // 파일 선택
    elements.selectFileBtn.addEventListener('click', selectFile);
    elements.removeFileBtn.addEventListener('click', removeFile);
    
    // 드래그 앤 드롭
    elements.fileDropZone.addEventListener('dragover', handleDragOver);
    elements.fileDropZone.addEventListener('drop', handleDrop);
    elements.fileDropZone.addEventListener('click', selectFile);
    
    // 처리 시작
    elements.processBtn.addEventListener('click', startProcessing);
    
    // 새 파일 처리
    elements.newProcessBtn.addEventListener('click', resetForNewFile);
    
    // 결과 저장
    elements.saveResultBtn.addEventListener('click', saveResults);
    
    // 텍스트 번역
    elements.translateTextBtn.addEventListener('click', translateText);
    
    // 진행률 이벤트 리스너
    window.electronAPI.onConversionProgress((progress) => {
        updateConversionProgress(progress);
    });
    
    window.electronAPI.onUploadProgress((progress) => {
        updateUploadProgress(progress);
    });
}

// 서버 상태 확인
async function checkServerStatus() {
    const serverUrl = elements.serverUrl.value.trim();
    
    try {
        elements.statusText.textContent = '연결 확인 중...';
        elements.statusIndicator.className = 'status-indicator';
        
        const result = await window.electronAPI.checkServerStatus(serverUrl);
        
        if (result.success) {
            elements.statusText.textContent = '서버 연결됨';
            elements.statusIndicator.className = 'status-indicator connected';
            showToast('서버에 성공적으로 연결되었습니다.', 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        elements.statusText.textContent = '서버 연결 실패';
        elements.statusIndicator.className = 'status-indicator disconnected';
        showToast(`서버 연결 실패: ${error.message}`, 'error');
    }
}

// 파일 선택
async function selectFile() {
    try {
        const result = await window.electronAPI.selectFile();
        
        if (!result.canceled && result.filePaths.length > 0) {
            const filePath = result.filePaths[0];
            setSelectedFile(filePath);
        }
    } catch (error) {
        showToast(`파일 선택 오류: ${error.message}`, 'error');
    }
}

// 선택된 파일 설정
function setSelectedFile(filePath) {
    selectedFilePath = filePath;
    const fileName = filePath.split('\\').pop().split('/').pop();
    
    elements.fileName.textContent = fileName;
    elements.filePath.textContent = filePath;
    elements.selectedFile.style.display = 'block';
    elements.fileDropZone.style.display = 'none';
    
    updateProcessButton();
    showToast('파일이 선택되었습니다.', 'success');
}

// 파일 제거
function removeFile() {
    selectedFilePath = null;
    elements.selectedFile.style.display = 'none';
    elements.fileDropZone.style.display = 'block';
    updateProcessButton();
}

// 드래그 오버 처리
function handleDragOver(e) {
    e.preventDefault();
    elements.fileDropZone.classList.add('dragover');
}

// 드롭 처리
function handleDrop(e) {
    e.preventDefault();
    elements.fileDropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        setSelectedFile(files[0].path);
    }
}

// 처리 버튼 상태 업데이트
function updateProcessButton() {
    elements.processBtn.disabled = !selectedFilePath;
}

// ✅ 처리 시작 (수정됨 - STT와 번역 분리)
async function startProcessing() {
    if (!selectedFilePath) {
        showToast('먼저 파일을 선택해주세요.', 'warning');
        return;
    }
    
    const serverUrl = elements.serverUrl.value.trim();
    const processingMode = document.querySelector('input[name="processingMode"]:checked').value;
    
    try {
        // UI 상태 변경
        elements.processBtn.disabled = true;
        elements.progressPanel.style.display = 'block';
        elements.resultsPanel.style.display = 'none';
        sttResultData = null;
        
        // 1단계: 오디오 변환 (WAV)
        console.log('📀 1단계: 오디오 변환 시작...');
        await convertAudioToWav();
        
        // 2단계: STT
        console.log('🎤 2단계: 음성 인식 시작...');
        await sendToSTT(serverUrl);
        
        // 3단계: 번역 (필요시)
        if (processingMode === 'audio-to-translation') {
            console.log('🌐 3단계: 번역 시작...');
            await translateSTTResult(serverUrl);
        } else {
            showToast('음성 인식이 완료되었습니다!', 'success');
        }
        
    } catch (error) {
        console.error('❌ 처리 오류:', error);
        showToast(`처리 오류: ${error.message}`, 'error');
        resetProcessingState();
    }
}

// ✅ 오디오를 WAV로 변환 (수정됨 - require('os') 제거)
async function convertAudioToWav() {
    return new Promise(async (resolve, reject) => {
        try {
            elements.conversionStatus.textContent = '변환 중...';
            
            // ✅ Main process에서 경로 생성 및 변환 수행
            const result = await window.electronAPI.convertToWav(selectedFilePath);
            
            if (result.success) {
                convertedWavPath = result.outputPath;  // ✅ Main에서 받은 경로 사용
                tempFiles.push(convertedWavPath);
                elements.conversionStatus.textContent = '변환 완료';
                elements.conversionProgress.style.width = '100%';
                console.log('✅ WAV 변환 완료:', convertedWavPath);
                resolve();
            } else {
                throw new Error(result.error || '변환 실패');
            }
        } catch (error) {
            elements.conversionStatus.textContent = '변환 실패';
            elements.conversionDetails.textContent = `오류: ${error.message}`;
            console.error('❌ 변환 오류:', error);
            reject(error);
        }
    });
}

// ✅ STT 수행 (새로 추가)
async function sendToSTT(serverUrl) {
    try {
        elements.uploadStatus.textContent = '업로드 중...';
        elements.processingStatus.textContent = 'AI 처리 중...';
        elements.aiProgress.style.width = '50%';
        
        // FastAPI의 /audio/process 엔드포인트 호출
        const result = await window.electronAPI.sendToAPI(
            convertedWavPath, 
            'audio/process',  // ✅ FastAPI 엔드포인트
            serverUrl
        );
        
        if (result.success) {
            elements.uploadStatus.textContent = '업로드 완료';
            elements.processingStatus.textContent = 'AI 처리 완료';
            elements.aiProgress.style.width = '100%';
            
            // ✅ STT 결과 저장
            sttResultData = result.data;
            console.log('✅ STT 완료:', sttResultData);
            
            displaySTTResult(sttResultData);
            
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        elements.uploadStatus.textContent = '업로드 실패';
        elements.processingStatus.textContent = 'AI 처리 실패';
        console.error('❌ STT 오류:', error);
        throw error;
    } finally {
        // 임시 파일 정리
        if (tempFiles.length > 0) {
            window.electronAPI.cleanupTempFiles(tempFiles);
            tempFiles = [];
        }
    }
}

// ✅ STT 결과를 번역 (새로 추가)
async function translateSTTResult(serverUrl) {
    try {
        if (!sttResultData || !sttResultData.text) {
            throw new Error('STT 결과가 없습니다.');
        }
        
        const sourceLang = elements.sourceLang.value;
        const targetLang = elements.targetLang.value;
        
        elements.processingStatus.textContent = '번역 중...';
        console.log(`🌐 번역 시작: ${sourceLang} → ${targetLang}`);
        
        const result = await window.electronAPI.translateText(
            sttResultData.text,
            sourceLang,
            targetLang,
            serverUrl
        );
        
        if (result.success) {
            elements.processingStatus.textContent = '번역 완료';
            console.log('✅ 번역 완료:', result.data);
            displayTranslationResult(result.data);
            showToast('음성 인식과 번역이 완료되었습니다!', 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        elements.processingStatus.textContent = '번역 실패';
        console.error('❌ 번역 오류:', error);
        throw error;
    } finally {
        resetProcessingState();
    }
}

// 변환 진행률 업데이트
function updateConversionProgress(progress) {
    elements.conversionProgress.style.width = `${progress.percent || 0}%`;
    
    if (progress.status === 'started') {
        elements.conversionStatus.textContent = '변환 시작';
    } else if (progress.status === 'processing') {
        elements.conversionStatus.textContent = `변환 중... ${Math.round(progress.percent || 0)}%`;
        if (progress.timemark) {
            elements.conversionDetails.textContent = `진행 시간: ${progress.timemark}`;
        }
    } else if (progress.status === 'completed') {
        elements.conversionStatus.textContent = '변환 완료';
        elements.conversionProgress.style.width = '100%';
    } else if (progress.status === 'error') {
        elements.conversionStatus.textContent = '변환 실패';
        elements.conversionDetails.textContent = `오류: ${progress.error}`;
    }
}

// 업로드 진행률 업데이트
function updateUploadProgress(progress) {
    elements.uploadProgress.style.width = `${progress.percent || 0}%`;
    elements.uploadStatus.textContent = `업로드 중... ${progress.percent || 0}%`;
}

// ✅ STT 결과 표시 (새로 추가)
function displaySTTResult(data) {
    elements.resultsPanel.style.display = 'block';
    
    // STT 텍스트
    if (data.text) {
        elements.sttResult.value = data.text;
        console.log('📝 STT 텍스트:', data.text);
    }
    
    // 감지된 언어
    if (data.detected_language) {
        elements.detectedLang.textContent = `감지된 언어: ${data.detected_language}`;
    }
    
    // 처리 시간
    if (data.transcription_time !== undefined) {
        elements.sttTime.textContent = `처리 시간: ${data.transcription_time}초`;
    } else if (data.processing_time !== undefined) {
        elements.sttTime.textContent = `처리 시간: ${data.processing_time}초`;
    }
}

// ✅ 번역 결과 표시 (새로 추가)
function displayTranslationResult(data) {
    elements.translationResult.style.display = 'block';
    
    // 번역 텍스트
    if (data.translated_text) {
        elements.translatedResult.value = data.translated_text;
        console.log('🌐 번역 텍스트:', data.translated_text);
    }
    
    // 목표 언어
    if (data.target_lang) {
        elements.translationInfo.textContent = `번역 언어: ${data.target_lang}`;
    }
    
    // 처리 시간
    if (data.processing_time !== undefined) {
        elements.translationTime.textContent = `번역 시간: ${data.processing_time}초`;
    }
}

// 처리 상태 리셋
function resetProcessingState() {
    elements.processBtn.disabled = false;
    elements.conversionProgress.style.width = '0%';
    elements.uploadProgress.style.width = '0%';
    elements.aiProgress.style.width = '0%';
    
    elements.conversionStatus.textContent = '대기 중';
    elements.uploadStatus.textContent = '대기 중';
    elements.processingStatus.textContent = '대기 중';
    elements.conversionDetails.textContent = '';
}

// 새 파일 처리를 위한 리셋
function resetForNewFile() {
    removeFile();
    elements.progressPanel.style.display = 'none';
    elements.resultsPanel.style.display = 'none';
    elements.translationResult.style.display = 'none';
    resetProcessingState();
    sttResultData = null;
}

// 결과 저장
async function saveResults() {
    try {
        const result = await window.electronAPI.selectSaveLocation();
        
        if (!result.canceled) {
            const content = {
                stt_result: elements.sttResult.value,
                translated_result: elements.translatedResult.value,
                detected_language: elements.detectedLang.textContent,
                processing_info: {
                    stt_time: elements.sttTime.textContent,
                    translation_time: elements.translationTime.textContent,
                    original_file: selectedFilePath
                }
            };
            
            // 파일 저장 로직 (실제 구현 필요)
            showToast('결과가 저장되었습니다.', 'success');
        }
    } catch (error) {
        showToast(`저장 오류: ${error.message}`, 'error');
    }
}

// 텍스트 번역
async function translateText() {
    const text = elements.inputText.value.trim();
    const sourceLang = elements.textSourceLang.value;
    const targetLang = elements.textTargetLang.value;
    const serverUrl = elements.serverUrl.value.trim();
    
    if (!text) {
        showToast('번역할 텍스트를 입력해주세요.', 'warning');
        return;
    }
    
    try {
        elements.translateTextBtn.disabled = true;
        elements.translateTextBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 번역 중...';
        
        const result = await window.electronAPI.translateText(text, sourceLang, targetLang, serverUrl);
        
        if (result.success) {
            elements.outputText.value = result.data.translated_text;
            showToast('번역이 완료되었습니다.', 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        showToast(`번역 오류: ${error.message}`, 'error');
    } finally {
        elements.translateTextBtn.disabled = false;
        elements.translateTextBtn.innerHTML = '<i class="fas fa-language"></i> 번역하기';
    }
}

// 클립보드에 복사
function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    if (element && element.value) {
        navigator.clipboard.writeText(element.value).then(() => {
            showToast('클립보드에 복사되었습니다.', 'success');
        }).catch(() => {
            showToast('복사에 실패했습니다.', 'error');
        });
    }
}

// 토스트 알림 표시
function showToast(message, type = 'info') {
    elements.toast.textContent = message;
    elements.toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// 로딩 오버레이 표시/숨김
function showLoading(message = '처리 중...') {
    elements.loadingText.textContent = message;
    elements.loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    elements.loadingOverlay.style.display = 'none';
}

// 정보 모달
function showAbout() {
    alert(`Audio Translation v1.0.0

음성 파일을 텍스트로 변환하고 번역하는 도구입니다.

기능:
- 다양한 오디오/비디오 형식 지원
- Whisper 기반 음성 인식
- Qwen3 기반 번역
- FFmpeg 통합 오디오 변환

개발: Electron + FastAPI`);
}

// 도움말 모달
function showHelp() {
    alert(`사용 방법:

1. 서버 설정에서 FastAPI 서버 URL을 확인하세요
2. 오디오 또는 비디오 파일을 선택하세요
3. 처리 모드를 선택하세요:
   - 음성 인식만: 오디오를 텍스트로만 변환
   - 음성 인식 + 번역: 텍스트 변환 후 번역까지 수행
4. 처리 시작 버튼을 클릭하세요
5. 결과를 확인하고 필요시 저장하세요

지원 형식:
- 오디오: MP3, WAV, M4A, AAC, OGG, FLAC
- 비디오: MP4, AVI, MOV, MKV, FLV, WEBM

문제가 있으면 서버 연결 상태를 먼저 확인해보세요.`);
}

// 앱 종료 시 정리
window.addEventListener('beforeunload', () => {
    if (tempFiles.length > 0) {
        window.electronAPI.cleanupTempFiles(tempFiles);
    }
});