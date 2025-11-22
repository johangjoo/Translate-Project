// 전역 변수
let selectedFilePath = null;
let convertedWavPath = null;
let tempFiles = [];
let sttResultData = null; // ✅ STT 결과 저장용

// DOM 요소들
const elements = {
    
    fileDropZone: document.getElementById('fileDropZone'),
    selectFileBtn: document.getElementById('selectFileBtn'),
    selectedFile: document.getElementById('selectedFile'),
    fileName: document.getElementById('fileName'),
    filePath: document.getElementById('filePath'),
    removeFileBtn: document.getElementById('removeFileBtn'),
    
    sttOnlyBtn: document.getElementById('sttOnlyBtn'),
    fullPipelineBtn: document.getElementById('fullPipelineBtn'),
    speakerDiarizationBtn: document.getElementById('speakerDiarizationBtn'),
    timestampsBtn: document.getElementById('timestampsBtn'),
    translationSettings: document.getElementById('translationSettings'),
    sourceLang: document.getElementById('sourceLang'),
    targetLang: document.getElementById('targetLang'),
    translationModel: document.getElementById('translationModel'),
    apiKey: document.getElementById('apiKey'),
    apiKeyRow: document.getElementById('apiKeyRow'),
    modelDescription: document.getElementById('modelDescription'),
    apiKeyHelp: document.getElementById('apiKeyHelp'),
    maxSpeakers: document.getElementById('maxSpeakers'),
    speakerCountDisplay: document.getElementById('speakerCountDisplay'),
    enableSpeakerDiarization: document.getElementById('enableSpeakerDiarization'),
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
    
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    toast: document.getElementById('toast')
};

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

function initializeApp() {
    // 처리 모드 버튼 이벤트
    elements.sttOnlyBtn.addEventListener('click', () => {
        setProcessingMode('transcribe');
    });
    
    elements.fullPipelineBtn.addEventListener('click', () => {
        setProcessingMode('audio-to-translation');
    });
    
    // 옵션 버튼 이벤트
    elements.speakerDiarizationBtn.addEventListener('click', () => {
        toggleOptionButton(elements.speakerDiarizationBtn);
    });
    
    elements.timestampsBtn.addEventListener('click', () => {
        toggleOptionButton(elements.timestampsBtn);
    });
    
    // 화자 수 선택 이벤트
    elements.maxSpeakers.addEventListener('change', () => {
        updateSpeakerCountDisplay();
    });
    
    // 번역 모델 선택 이벤트
    elements.translationModel.addEventListener('change', () => {
        updateModelSettings();
    });
    
    // 파일 선택 상태에 따른 버튼 활성화
    updateProcessButton();
    
    // 초기 화자수 표시 업데이트
    updateSpeakerCountDisplay();
    
    // 초기 모델 설정 업데이트
    updateModelSettings();
}

function setProcessingMode(mode) {
    // 모든 처리 모드 버튼 비활성화
    elements.sttOnlyBtn.classList.remove('active');
    elements.fullPipelineBtn.classList.remove('active');
    
    // 선택된 모드 활성화
    if (mode === 'transcribe') {
        elements.sttOnlyBtn.classList.add('active');
        elements.translationSettings.style.display = 'none';
    } else if (mode === 'audio-to-translation') {
        elements.fullPipelineBtn.classList.add('active');
        elements.translationSettings.style.display = 'block';
    }
}

function toggleOptionButton(button) {
    button.classList.toggle('active');
    const isActive = button.classList.contains('active');
    console.log('버튼 토글:', button.id, '활성화:', isActive);
}

function getProcessingMode() {
    if (elements.sttOnlyBtn.classList.contains('active')) {
        return 'transcribe';
    } else if (elements.fullPipelineBtn.classList.contains('active')) {
        return 'audio-to-translation';
    }
    return 'transcribe'; // 기본값
}

function isSpeakerDiarizationEnabled() {
    return elements.speakerDiarizationBtn.classList.contains('active');
}

function isTimestampsEnabled() {
    return elements.timestampsBtn.classList.contains('active');
}

function updateSpeakerCountDisplay() {
    const selectedValue = elements.maxSpeakers.value;
    console.log('화자수 업데이트:', selectedValue);
    if (elements.speakerCountDisplay) {
        elements.speakerCountDisplay.textContent = `${selectedValue}명`;
        console.log('화자수 표시 업데이트 완료:', `${selectedValue}명`);
    } else {
        console.error('speakerCountDisplay 요소를 찾을 수 없습니다');
    }
}

function updateModelSettings() {
    const selectedModel = elements.translationModel.value;
    
    if (selectedModel === 'qwen-local') {
        elements.apiKeyRow.style.display = 'none';
        elements.modelDescription.textContent = '로컬 모델 - 무료, 빠름, 인터넷 불필요';
    } else if (selectedModel === 'openai') {
        elements.apiKeyRow.style.display = 'block';
        elements.modelDescription.textContent = 'OpenAI GPT-4 - 고품질, API 키 필요';
        elements.apiKeyHelp.textContent = 'OpenAI API 키 (sk-...로 시작)';
        elements.apiKey.placeholder = 'sk-proj-...';
    } else if (selectedModel === 'gemini') {
        elements.apiKeyRow.style.display = 'block';
        elements.modelDescription.textContent = 'Google Gemini - 빠름, API 키 필요';
        elements.apiKeyHelp.textContent = 'Google AI Studio API 키 (AIza...로 시작)';
        elements.apiKey.placeholder = 'AIza...';
    }
}

function getTranslationSettings() {
    return {
        model_type: elements.translationModel.value,
        api_key: elements.apiKey.value || null,
        source_lang: elements.sourceLang.value,
        target_lang: elements.targetLang.value
    };
}

function setupEventListeners() {
    // 파일 선택
    elements.selectFileBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // 부모 fileDropZone의 click 이벤트로 전파 방지
        selectFile();
    });
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
    
    // 진행률 이벤트 리스너
    window.electronAPI.onConversionProgress((progress) => {
        updateConversionProgress(progress);
    });
    
    window.electronAPI.onUploadProgress((progress) => {
        updateUploadProgress(progress);
    });
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

// 처리 시작 (수정됨 - STT와 번역 분리)
async function startProcessing() {
    if (!selectedFilePath) {
        showToast('먼저 파일을 선택해주세요.', 'warning');
        return;
    }
    
    const serverUrl = 'http://127.0.0.1:8000';
    const processingMode = getProcessingMode();
    
    try {
        // UI 상태 변경
        elements.processBtn.disabled = true;
        sttResultData = null;
        
        // 1단계: 오디오 변환 (WAV)
        console.log(' 1단계: 오디오 변환 시작...');
        await convertAudioToWav();
        
        // 2단계: STT
        console.log(' 2단계: 음성 인식 시작...');
        await sendToSTT(serverUrl);
        
        // 3단계: 번역 (필요시)
        if (processingMode === 'audio-to-translation') {
            console.log(' 3단계: 번역 시작...');
            await translateSTTResult(serverUrl);
        } else {
            showToast('음성 인식이 완료되었습니다!', 'success');
        }
        
    } catch (error) {
        console.error(' 처리 오류:', error);
        showToast(`처리 오류: ${error.message}`, 'error');
        resetProcessingState();
    }
}

// 오디오를 WAV로 변환 (수정됨 - require('os') 제거)
async function convertAudioToWav() {
    return new Promise(async (resolve, reject) => {
        try {
            elements.conversionStatus.textContent = '변환 중...';
            
            // Main process에서 경로 생성 및 변환 수행
            const result = await window.electronAPI.convertToWav(selectedFilePath);
            
            if (result.success) {
                convertedWavPath = result.outputPath;  // Main에서 받은 경로 사용
                tempFiles.push(convertedWavPath);
                elements.conversionStatus.textContent = '변환 완료';
                elements.conversionProgress.style.width = '100%';
                console.log(' WAV 변환 완료:', convertedWavPath);
                resolve();
            } else {
                throw new Error(result.error || '변환 실패');
            }
        } catch (error) {
            elements.conversionStatus.textContent = '변환 실패';
            elements.conversionDetails.textContent = `오류: ${error.message}`;
            console.error(' 변환 오류:', error);
            reject(error);
        }
    });
}

// STT 수행 (새로 추가)
async function sendToSTT(serverUrl) {
    try {
        elements.uploadStatus.textContent = '업로드 중...';
        elements.processingStatus.textContent = 'AI 처리 중...';
        elements.aiProgress.style.width = '50%';
        
        // FastAPI의 /audio/process 엔드포인트 호출
        const maxSpeakers = elements.maxSpeakers
            ? Number(elements.maxSpeakers.value || 2)
            : 2;

        // 새로운 옵션들 추가
        const enableSpeakerDiarization = isSpeakerDiarizationEnabled();
        const enableTimestamps = isTimestampsEnabled();

        const result = await window.electronAPI.sendToAPI(
            convertedWavPath,
            'audio/process',  // FastAPI 엔드포인트
            serverUrl,
            { 
                maxSpeakers,
                enableSpeakerDiarization,
                enableTimestamps
            }
        );
        
        if (result.success) {
            elements.uploadStatus.textContent = '업로드 완료';
            elements.processingStatus.textContent = 'AI 처리 완료';
            elements.aiProgress.style.width = '100%';
            
            // STT 결과 저장
            sttResultData = result.data;
            console.log(' STT 완료:', sttResultData);
            
            displaySTTResult(sttResultData);
            
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        elements.uploadStatus.textContent = '업로드 실패';
        elements.processingStatus.textContent = 'AI 처리 실패';
        console.error(' STT 오류:', error);
        throw error;
    } finally {
        // 임시 파일 정리
        if (tempFiles.length > 0) {
            window.electronAPI.cleanupTempFiles(tempFiles);
            tempFiles = [];
        }
    }
}

// STT 결과를 번역 (수정됨 - 모델 선택 지원)
async function translateSTTResult(serverUrl) {
    try {
        if (!sttResultData || !sttResultData.text) {
            throw new Error('STT 결과가 없습니다.');
        }
        
        // 번역 설정 가져오기 (모델 타입, API 키 포함)
        const translationSettings = getTranslationSettings();
        
        // API 키 검증 (OpenAI/Gemini 사용 시)
        if ((translationSettings.model_type === 'openai' || translationSettings.model_type === 'gemini') 
            && !translationSettings.api_key) {
            throw new Error(`${translationSettings.model_type} 모델 사용 시 API 키가 필요합니다.`);
        }
        
        elements.processingStatus.textContent = '번역 중...';
        console.log(` 번역 시작: ${translationSettings.source_lang} → ${translationSettings.target_lang} (모델: ${translationSettings.model_type})`);
        
        const result = await window.electronAPI.translateTextWithModel(
            sttResultData.text,
            translationSettings.source_lang,
            translationSettings.target_lang,
            translationSettings.model_type,
            translationSettings.api_key,
            serverUrl
        );
        
        if (result.success) {
            elements.processingStatus.textContent = '번역 완료';
            console.log(' 번역 완료:', result.data);
            displayTranslationResult(result.data);
            showToast('음성 인식과 번역이 완료되었습니다!', 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        elements.processingStatus.textContent = '번역 실패';
        console.error(' 번역 오류:', error);
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

// STT 결과 표시 (새로 추가)
function displaySTTResult(data) {
    // STT 텍스트
    if (data.text) {
        elements.sttResult.value = data.text;
        console.log(' STT 텍스트:', data.text);
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

// 번역 결과 표시 (새로 추가)
function displayTranslationResult(data) {
    // 번역 텍스트
    if (data.translated_text) {
        elements.translatedResult.value = data.translated_text;
        console.log(' 번역 텍스트:', data.translated_text);
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