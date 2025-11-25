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
    sttLanguageSettings: document.getElementById('sttLanguageSettings'),
    sttLanguage: document.getElementById('sttLanguage'),
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
    saveTxtBtn: document.getElementById('saveTxtBtn'),
    saveSrtBtn: document.getElementById('saveSrtBtn'),
    
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
        updateSrtButtonState(); // 타임스탬프 버튼 클릭 시 SRT 버튼 상태 업데이트
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
    
    // 초기 처리 모드 설정 (음성인식만이 기본값)
    setProcessingMode('transcribe');
    
    // 초기 SRT 버튼 상태 설정
    updateSrtButtonState();
}

function setProcessingMode(mode) {
    // 모든 처리 모드 버튼 비활성화
    elements.sttOnlyBtn.classList.remove('active');
    elements.fullPipelineBtn.classList.remove('active');
    
    // 선택된 모드 활성화
    if (mode === 'transcribe') {
        elements.sttOnlyBtn.classList.add('active');
        elements.sttLanguageSettings.style.display = 'block';
        elements.translationSettings.style.display = 'none';
    } else if (mode === 'audio-to-translation') {
        elements.fullPipelineBtn.classList.add('active');
        elements.sttLanguageSettings.style.display = 'none';
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

// SRT 버튼 상태 업데이트 (타임스탬프 활성화 여부에 따라)
function updateSrtButtonState() {
    const timestampsEnabled = isTimestampsEnabled();
    const hasSttResult = sttResultData !== null;
    
    // 타임스탬프가 비활성화되어 있거나 STT 결과가 없으면 SRT 버튼 비활성화
    if (!timestampsEnabled || !hasSttResult) {
        elements.saveSrtBtn.disabled = true;
        elements.saveSrtBtn.style.opacity = '0.5';
        elements.saveSrtBtn.style.cursor = 'not-allowed';
    } else {
        elements.saveSrtBtn.disabled = false;
        elements.saveSrtBtn.style.opacity = '1';
        elements.saveSrtBtn.style.cursor = 'pointer';
    }
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
        elements.modelDescription.textContent = 'Qwen3는 무료 모델이지만 번역 성능이 보통입니다, API키는 필요하지 않습니다.';
    } else if (selectedModel === 'openai') {
        elements.apiKeyRow.style.display = 'block';
        elements.modelDescription.textContent = 'OpenAI GPT-5.1 모델은 고품질 번역을 제공하지만 유료 API키가 필요합니다.';
        elements.apiKeyHelp.textContent = 'OpenAI API 키 (sk-...로 시작)';
        elements.apiKey.placeholder = 'sk-proj-...';
    } else if (selectedModel === 'gemini') {
        elements.apiKeyRow.style.display = 'block';
        elements.modelDescription.textContent = 'Google Gemini 2.5 flash 모델은 고품질 번역을 제공하지만 API키가 필요합니다.';
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
    elements.saveTxtBtn.addEventListener('click', saveResultsTxt);
    elements.saveSrtBtn.addEventListener('click', saveResultsSrt);
    
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
        // 결과 영역 비우기
        clearResults();
        
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
        console.log('처리 모드 확인:', processingMode);
        if (processingMode === 'audio-to-translation') {
            console.log(' 3단계: 번역 시작...');
            await translateSTTResult(serverUrl);
        } else {
            console.log('번역 모드가 아니므로 번역 스킵. 모드:', processingMode);
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
        
        // STT 언어 설정 (음성인식만 모드일 때)
        const sttLanguage = elements.sttLanguage ? elements.sttLanguage.value : null;
        const language = sttLanguage && sttLanguage !== '' ? sttLanguage : null;

        const result = await window.electronAPI.sendToAPI(
            convertedWavPath,
            'audio/process',  // FastAPI 엔드포인트
            serverUrl,
            { 
                maxSpeakers,
                enableSpeakerDiarization,
                enableTimestamps,
                language: language  // 언어 파라미터 추가
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
            
            // SRT 버튼 상태 업데이트 (타임스탬프 활성화 여부 확인)
            updateSrtButtonState();
            
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
        console.log('translateSTTResult 호출됨, sttResultData:', sttResultData);
        if (!sttResultData) {
            throw new Error('STT 결과 데이터가 없습니다.');
        }
        
        // STT 결과에서 텍스트 추출
        let textToTranslate = sttResultData.text || sttResultData.simple_text || '';
        if (!textToTranslate && sttResultData.segments) {
            // 세그먼트에서 텍스트 추출
            textToTranslate = sttResultData.segments.map(s => s.text || s.transcript || '').join('\n');
        }
        
        if (!textToTranslate || textToTranslate.trim() === '') {
            console.error('번역할 텍스트가 없습니다. sttResultData:', sttResultData);
            throw new Error('STT 결과에서 번역할 텍스트를 찾을 수 없습니다.');
        }
        
        console.log('번역할 텍스트:', textToTranslate.substring(0, 100) + '...');
        
        // 번역 설정 가져오기 (모델 타입, API 키 포함)
        const translationSettings = getTranslationSettings();
        
        // API 키 검증 (OpenAI/Gemini 사용 시)
        if ((translationSettings.model_type === 'openai' || translationSettings.model_type === 'gemini') 
            && !translationSettings.api_key) {
            throw new Error(`${translationSettings.model_type} 모델 사용 시 API 키가 필요합니다.`);
        }
        
        elements.processingStatus.textContent = '번역 중...';
        console.log(` 번역 시작: ${translationSettings.source_lang} → ${translationSettings.target_lang} (모델: ${translationSettings.model_type})`);
        
        const result = await window.electronAPI.translateText(
            textToTranslate,
            translationSettings.source_lang,
            translationSettings.target_lang,
            serverUrl,
            translationSettings.model_type,
            translationSettings.api_key
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

// 결과 영역 비우기
function clearResults() {
    // 음성 인식 결과 비우기
    elements.sttResult.value = '';
    elements.detectedLang.textContent = '';
    elements.sttTime.textContent = '';
    
    // 번역 결과 비우기
    elements.translatedResult.value = '';
    elements.translationInfo.textContent = '';
    elements.translationTime.textContent = '';
}

// 새 파일 처리를 위한 리셋
function resetForNewFile() {
    // 결과 영역 비우기
    clearResults();
    
    removeFile();
    resetProcessingState();
    sttResultData = null;
    
    // SRT 버튼 상태 업데이트
    updateSrtButtonState();
}

// 결과 저장
// TXT 파일로 저장
async function saveResultsTxt() {
    try {
        if (!elements.sttResult.value && !elements.translatedResult.value) {
            showToast('저장할 내용이 없습니다.', 'error');
            return;
        }

        const result = await window.electronAPI.selectSaveTxtLocation();
        
        if (!result.canceled && result.filePath) {
            // 저장할 내용 구성
            let content = '';
            
            // 음성 인식 결과
            if (elements.sttResult.value) {
                content += '=== 음성 인식 결과 ===\n';
                content += `감지된 언어: ${elements.detectedLang.textContent || '알 수 없음'}\n`;
                content += `처리 시간: ${elements.sttTime.textContent || '알 수 없음'}\n\n`;
                content += elements.sttResult.value + '\n\n';
            }
            
            // 번역 결과
            if (elements.translatedResult.value) {
                content += '=== 번역 결과 ===\n';
                content += `번역 정보: ${elements.translationInfo.textContent || '알 수 없음'}\n`;
                content += `처리 시간: ${elements.translationTime.textContent || '알 수 없음'}\n\n`;
                content += elements.translatedResult.value + '\n';
            }
            
            // 원본 파일 정보
            if (selectedFilePath) {
                content += `\n원본 파일: ${selectedFilePath}\n`;
            }
            
            // 파일 저장
            await window.electronAPI.saveTxtFile(result.filePath, content);
            showToast('TXT 파일로 저장되었습니다.', 'success');
        }
    } catch (error) {
        showToast(`저장 오류: ${error.message}`, 'error');
    }
}

// SRT 파일로 저장
async function saveResultsSrt() {
    try {
        if (!sttResultData || !sttResultData.text) {
            showToast('저장할 STT 결과가 없습니다.', 'error');
            return;
        }

        const result = await window.electronAPI.selectSaveSrtLocation();
        
        if (!result.canceled && result.filePath) {
            // 번역 결과가 있으면 번역 텍스트 사용, 없으면 원본 STT 사용
            const hasTranslation = elements.translatedResult && elements.translatedResult.value.trim();
            let srtContent;
            
            if (hasTranslation) {
                // 번역 텍스트를 사용하여 SRT 생성 (원본 타임스탬프 유지)
                srtContent = convertSttDataToSrtWithTranslation(sttResultData, elements.translatedResult.value);
            } else {
                // 원본 STT 텍스트 사용
                srtContent = convertSttDataToSrt(sttResultData);
            }
            
            if (!srtContent) {
                showToast('SRT 파일을 생성할 수 없습니다. 세그먼트 정보가 없습니다.', 'error');
                return;
            }
            
            // 파일 저장
            await window.electronAPI.saveSrtFile(result.filePath, srtContent);
            showToast('SRT 파일로 저장되었습니다.', 'success');
        }
    } catch (error) {
        showToast(`저장 오류: ${error.message}`, 'error');
    }
}

// STT 데이터를 SRT 형식으로 변환
function convertSttDataToSrt(sttData) {
    // 설정
    const MAX_DURATION = 5.0;  // 자막이 화면에 떠 있는 최대 시간(초)
    const REMOVE_SPEAKER = false;  // True로 설정하면 "화자A: " 부분을 지우고 내용만 남김
    
    // 방법 1: segments 배열이 있는 경우
    if (sttData.segments && Array.isArray(sttData.segments) && sttData.segments.length > 0) {
        return convertSegmentsToSrt(sttData.segments, MAX_DURATION, REMOVE_SPEAKER);
    }
    
    // 방법 2: simple_text 형식의 텍스트가 있는 경우 ([00:00.000] 화자A: 텍스트)
    if (sttData.simple_text || sttData.text) {
        const logText = sttData.simple_text || sttData.text;
        return convertFullLogToSrt(logText, MAX_DURATION, REMOVE_SPEAKER);
    }
    
    return null;
}

// STT 데이터를 번역 텍스트와 함께 SRT 형식으로 변환
function convertSttDataToSrtWithTranslation(sttData, translatedText) {
    const MAX_DURATION = 5.0;
    const REMOVE_SPEAKER = false;
    
    // 번역 텍스트 정리 (타임스탬프 제거)
    const cleanedTranslatedText = cleanTranslatedText(translatedText);
    
    // segments 배열이 있는 경우
    if (sttData.segments && Array.isArray(sttData.segments) && sttData.segments.length > 0) {
        // 원본 세그먼트 텍스트 길이 비율에 따라 번역 텍스트 분할
        const translatedSegments = splitTranslatedTextByOriginalSegments(cleanedTranslatedText, sttData.segments);
        
        const srtOutput = [];
        for (let i = 0; i < sttData.segments.length; i++) {
            const segment = sttData.segments[i];
            const startTime = parseFloat(segment.start || 0);
            // 분할된 번역 텍스트 사용
            let content = translatedSegments[i] ? translatedSegments[i].trim() : '';
            
            // 번역 텍스트가 없으면 원본 세그먼트 텍스트 사용 (fallback)
            // 단, 마지막 세그먼트인 경우 번역 텍스트 전체에서 마지막 부분 추출 시도
            if (!content) {
                if (i === sttData.segments.length - 1 && cleanedTranslatedText) {
                    // 마지막 세그먼트인 경우 번역 텍스트의 마지막 부분 사용
                    const words = cleanedTranslatedText.split(' ').filter(w => w.trim());
                    if (words.length > 0) {
                        // 마지막 몇 단어 사용 (원본 세그먼트 길이에 비례)
                        const originalText = (segment.text || segment.transcript || '').trim();
                        const originalTextClean = originalText.replace(/^[^:]+:\s*/, '');
                        const wordCount = Math.max(1, Math.ceil(originalTextClean.split(' ').length));
                        const lastWords = words.slice(-wordCount).join(' ');
                        content = lastWords;
                    }
                }
                
                // 여전히 없으면 원본 텍스트 사용
                if (!content) {
                    const originalText = (segment.text || segment.transcript || '').trim();
                    // 원본 텍스트에서 화자 정보 제거
                    content = originalText.replace(/^[^:]+:\s*/, '');
                }
            }
            
            // 화자 정보 처리
            if (segment.speaker && !REMOVE_SPEAKER) {
                if (!content.includes(segment.speaker)) {
                    content = `${segment.speaker}: ${content}`;
                }
            } else if (REMOVE_SPEAKER && segment.speaker) {
                content = content.replace(/^[^:]+:\s*/, '');
            }
            
            // 종료 시간 계산
            let endTime;
            if (i < sttData.segments.length - 1) {
                const nextStart = parseFloat(sttData.segments[i + 1].start || startTime);
                endTime = Math.min(nextStart, startTime + MAX_DURATION);
                if (endTime <= startTime) {
                    endTime = startTime + 2.0;
                }
            } else {
                const duration = parseFloat(segment.end || segment.start || 0) - startTime;
                endTime = startTime + Math.max(duration, 2.0);
            }
            
            const srtBlock = `${i + 1}\n` +
                `${secondsToSrtTimestamp(startTime)} --> ${secondsToSrtTimestamp(endTime)}\n` +
                `${content}\n\n`;
            srtOutput.push(srtBlock);
        }
        return srtOutput.join('');
    }
    
    // segments가 없으면 simple_text 형식 파싱
    if (sttData.simple_text || sttData.text) {
        const logText = sttData.simple_text || sttData.text;
        const parsedLines = parseLogTextForSrt(logText);
        if (parsedLines.length > 0) {
            // 원본 세그먼트 텍스트 길이 비율에 따라 번역 텍스트 분할
            const translatedSegments = splitTranslatedTextByOriginalLines(cleanedTranslatedText, parsedLines);
            
            const srtOutput = [];
            for (let i = 0; i < parsedLines.length; i++) {
                const current = parsedLines[i];
                const startTime = current.start;
                // 분할된 번역 텍스트 사용
                let content = translatedSegments[i] ? translatedSegments[i].trim() : '';
                
                // 번역 텍스트가 없으면 원본 텍스트 사용 (fallback)
                if (!content) {
                    // 원본 텍스트에서 화자 정보 제거
                    content = current.content.replace(/^[^:]+:\s*/, '');
                }
                
                // 종료 시간 계산
                let endTime;
                if (i < parsedLines.length - 1) {
                    const nextStart = parsedLines[i + 1].start;
                    endTime = Math.min(nextStart, startTime + MAX_DURATION);
                    if (endTime <= startTime) {
                        endTime = startTime + 2.0;
                    }
                } else {
                    endTime = startTime + 2.0;
                }
                
                const srtBlock = `${i + 1}\n` +
                    `${secondsToSrtTimestamp(startTime)} --> ${secondsToSrtTimestamp(endTime)}\n` +
                    `${content}\n\n`;
                srtOutput.push(srtBlock);
            }
            return srtOutput.join('');
        }
    }
    
    return null;
}

// 번역 텍스트 정리 (타임스탬프, 화자 정보 제거)
function cleanTranslatedText(translatedText) {
    if (!translatedText) return '';
    
    // 타임스탬프 패턴 제거 [00:00.000]
    let cleaned = translatedText.replace(/\[\d{2}:\d{2}\.\d{3}\]/g, '');
    
    // 화자 정보 제거 (화자A:, 화자B: 등)
    cleaned = cleaned.replace(/화자[가-힣A-Z]:\s*/g, '');
    
    // 여러 공백을 하나로
    cleaned = cleaned.replace(/\s+/g, ' ').trim();
    
    return cleaned;
}

// 원본 세그먼트의 텍스트 길이 비율에 따라 번역 텍스트 분할
function splitTranslatedTextByOriginalSegments(translatedText, originalSegments) {
    if (!translatedText || translatedText.trim() === '' || !originalSegments || originalSegments.length === 0) {
        return new Array(originalSegments.length).fill('');
    }
    
    if (originalSegments.length === 1) {
        return [translatedText.trim()];
    }
    
    const trimmedText = translatedText.trim();
    
    // 원본 세그먼트의 텍스트 길이 계산 (화자 정보 제거)
    const originalTexts = originalSegments.map(seg => {
        const text = (seg.text || seg.transcript || '').trim();
        return text.replace(/^[^:]+:\s*/, ''); // 화자 정보 제거
    });
    
    const originalLengths = originalTexts.map(text => text.length);
    const totalOriginalLength = originalLengths.reduce((sum, len) => sum + len, 0);
    
    if (totalOriginalLength === 0) {
        // 원본 텍스트 길이를 계산할 수 없으면 균등 분할
        return splitTranslatedTextToSegments(translatedText, originalSegments.length);
    }
    
    // 원본 세그먼트의 길이 비율에 따라 번역 텍스트 분할
    const translatedSegments = [];
    let currentPos = 0;
    
    for (let i = 0; i < originalSegments.length; i++) {
        // 마지막 세그먼트는 남은 모든 텍스트 사용
        if (i === originalSegments.length - 1) {
            // currentPos가 텍스트 길이를 넘지 않도록 보장
            const startPos = Math.min(currentPos, trimmedText.length);
            const segmentText = trimmedText.substring(startPos).trim();
            translatedSegments.push(segmentText || '');
        } else {
            const ratio = originalLengths[i] / totalOriginalLength;
            const segmentLength = Math.round(trimmedText.length * ratio);
            
            // 단어 중간에서 자르지 않도록 공백 기준으로 조정
            let endPos = currentPos + segmentLength;
            
            // endPos가 텍스트 길이를 넘지 않도록 제한
            if (endPos >= trimmedText.length) {
                // 마지막 세그먼트가 아니므로 현재 위치에서 남은 텍스트의 일부만 사용
                endPos = Math.min(endPos, trimmedText.length - 1);
            }
            
            // 단어 중간이면 다음 공백까지 확장
            if (endPos < trimmedText.length && trimmedText[endPos] !== ' ') {
                const nextSpace = trimmedText.indexOf(' ', endPos);
                if (nextSpace !== -1) {
                    endPos = nextSpace;
                } else {
                    // 공백이 없으면 텍스트 끝까지
                    endPos = trimmedText.length;
                }
            }
            
            const segmentText = trimmedText.substring(currentPos, endPos).trim();
            translatedSegments.push(segmentText || '');
            currentPos = endPos;
        }
    }
    
    // 마지막 세그먼트가 비어있으면 이전 세그먼트에서 텍스트 가져오기
    if (translatedSegments.length > 0 && translatedSegments[translatedSegments.length - 1] === '') {
        // 마지막 세그먼트가 비어있으면 번역 텍스트의 마지막 부분 사용
        const lastSegmentText = trimmedText.split(' ').slice(-3).join(' ').trim(); // 마지막 3단어
        if (lastSegmentText) {
            translatedSegments[translatedSegments.length - 1] = lastSegmentText;
        }
    }
    
    return translatedSegments;
}

// 번역 텍스트를 세그먼트 수에 맞춰 분할 (fallback용)
function splitTranslatedTextToSegments(translatedText, segmentCount) {
    if (!translatedText || translatedText.trim() === '') {
        return new Array(segmentCount).fill('');
    }
    
    if (segmentCount <= 1) {
        return [translatedText.trim()];
    }
    
    const trimmedText = translatedText.trim();
    
    // 문장 단위로 분할 시도 (더 정확한 패턴)
    const sentencePattern = /[.!?。！？]\s*/;
    const sentences = trimmedText.split(sentencePattern).filter(s => s.trim());
    
    if (sentences.length >= segmentCount) {
        // 문장 수가 세그먼트 수보다 많거나 같으면 균등 분할
        const segments = [];
        const chunkSize = Math.ceil(sentences.length / segmentCount);
        
        for (let i = 0; i < segmentCount; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, sentences.length);
            
            if (start >= sentences.length) {
                segments.push(''); // 빈 문자열 (중복 방지)
            } else {
                const segmentText = sentences.slice(start, end)
                    .map(s => s.trim())
                    .filter(s => s.length > 0)
                    .join('. ');
                segments.push(segmentText || '');
            }
        }
        return segments;
    } else if (sentences.length > 0) {
        // 문장 수가 적으면 문장을 세그먼트에 균등 분배
        const segments = [];
        for (let i = 0; i < segmentCount; i++) {
            if (i < sentences.length) {
                segments.push(sentences[i].trim());
            } else {
                segments.push(''); // 빈 문자열
            }
        }
        return segments;
    } else {
        // 문장 구분이 안 되면 텍스트 길이로 균등 분할
        const segments = [];
        const chunkSize = Math.ceil(trimmedText.length / segmentCount);
        
        for (let i = 0; i < segmentCount; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, trimmedText.length);
            
            if (start >= trimmedText.length) {
                segments.push(''); // 빈 문자열
            } else {
                // 단어 중간에서 자르지 않도록 공백 기준으로 조정
                let segmentText = trimmedText.substring(start, end);
                if (end < trimmedText.length && trimmedText[end] !== ' ') {
                    // 단어 중간이면 다음 공백까지 확장
                    const nextSpace = trimmedText.indexOf(' ', end);
                    if (nextSpace !== -1) {
                        segmentText = trimmedText.substring(start, nextSpace);
                    }
                }
                segments.push(segmentText.trim() || '');
            }
        }
        return segments;
    }
}

// 원본 라인 텍스트 길이 비율에 따라 번역 텍스트 분할 (simple_text 형식용)
function splitTranslatedTextByOriginalLines(translatedText, originalLines) {
    if (!translatedText || translatedText.trim() === '' || !originalLines || originalLines.length === 0) {
        return new Array(originalLines.length).fill('');
    }
    
    if (originalLines.length === 1) {
        return [translatedText.trim()];
    }
    
    const trimmedText = translatedText.trim();
    
    // 원본 라인의 텍스트 길이 계산 (화자 정보 제거)
    const originalTexts = originalLines.map(line => {
        const text = line.content || '';
        return text.replace(/^[^:]+:\s*/, ''); // 화자 정보 제거
    });
    
    const originalLengths = originalTexts.map(text => text.length);
    const totalOriginalLength = originalLengths.reduce((sum, len) => sum + len, 0);
    
    if (totalOriginalLength === 0) {
        // 원본 텍스트 길이를 계산할 수 없으면 균등 분할
        return splitTranslatedTextToSegments(translatedText, originalLines.length);
    }
    
    // 원본 라인의 길이 비율에 따라 번역 텍스트 분할
    const translatedSegments = [];
    let currentPos = 0;
    
    for (let i = 0; i < originalLines.length; i++) {
        const ratio = originalLengths[i] / totalOriginalLength;
        const segmentLength = Math.round(trimmedText.length * ratio);
        
        // 마지막 라인은 남은 모든 텍스트 사용
        if (i === originalLines.length - 1) {
            const segmentText = trimmedText.substring(currentPos).trim();
            translatedSegments.push(segmentText || '');
        } else {
            // 단어 중간에서 자르지 않도록 공백 기준으로 조정
            let endPos = currentPos + segmentLength;
            
            // 단어 중간이면 다음 공백까지 확장
            if (endPos < trimmedText.length && trimmedText[endPos] !== ' ') {
                const nextSpace = trimmedText.indexOf(' ', endPos);
                if (nextSpace !== -1) {
                    endPos = nextSpace;
                }
            }
            
            const segmentText = trimmedText.substring(currentPos, endPos).trim();
            translatedSegments.push(segmentText || '');
            currentPos = endPos;
        }
    }
    
    return translatedSegments;
}

// 로그 텍스트 파싱 헬퍼 (SRT용)
function parseLogTextForSrt(logText) {
    const pattern = /\[(\d{2}:\d{2}\.\d{3})\](.*)/;
    const lines = logText.trim().split('\n');
    const parsedLines = [];
    
    for (const line of lines) {
        const match = line.match(pattern);
        if (match) {
            const timeStr = match[1];
            const content = match[2].trim();
            const startSeconds = timeStrToSeconds(timeStr);
            parsedLines.push({ start: startSeconds, content: content });
        }
    }
    
    return parsedLines;
}

// segments 배열을 SRT로 변환
function convertSegmentsToSrt(segments, maxDuration, removeSpeaker) {
    const srtOutput = [];
    
    for (let i = 0; i < segments.length; i++) {
        const current = segments[i];
        const startTime = parseFloat(current.start || 0);
        let content = current.text || current.transcript || '';
        
        // 화자 정보 처리
        if (current.speaker) {
            if (removeSpeaker) {
                // 화자 이름 제거
                content = content.replace(/^[^:]+:\s*/, '');
            } else {
                // 화자 이름 유지 (이미 content에 포함되어 있을 수 있음)
                if (!content.includes(current.speaker)) {
                    content = `${current.speaker}: ${content}`;
                }
            }
        }
        
        // 종료 시간 계산
        let endTime;
        if (i < segments.length - 1) {
            const nextStart = parseFloat(segments[i + 1].start || startTime);
            // 다음 자막 시작 시간과 (현재시간 + 최대지속시간) 중 더 짧은 것을 선택
            endTime = Math.min(nextStart, startTime + maxDuration);
            
            // 예외처리: 다음 자막이 현재보다 빠르거나 같으면 2초 더함
            if (endTime <= startTime) {
                endTime = startTime + 2.0;
            }
        } else {
            // 마지막 줄 처리
            const duration = parseFloat(current.end || current.start || 0) - startTime;
            endTime = startTime + Math.max(duration, 2.0);
        }
        
        // SRT 블록 조립
        const srtBlock = `${i + 1}\n` +
            `${secondsToSrtTimestamp(startTime)} --> ${secondsToSrtTimestamp(endTime)}\n` +
            `${content}\n\n`;
        
        srtOutput.push(srtBlock);
    }
    
    return srtOutput.join('');
}

// 로그 텍스트를 SRT로 변환 ([00:00.000] 화자A: 텍스트 형식)
function convertFullLogToSrt(logText, maxDuration, removeSpeaker) {
    const srtOutput = [];
    const parsedLines = [];
    
    // 정규표현식: [00:00.000] 형태만 정확히 캐치
    const pattern = /\[(\d{2}:\d{2}\.\d{3})\](.*)/;
    
    // 1. 텍스트를 줄 단위로 나누고 파싱
    const lines = logText.trim().split('\n');
    
    for (const line of lines) {
        const match = line.match(pattern);
        if (match) {
            const timeStr = match[1];
            let content = match[2].trim();
            
            // 옵션: 화자 이름 제거 (ex: "화자B: 안녕" -> "안녕")
            if (removeSpeaker) {
                if (content.includes(':')) {
                    content = content.split(':').slice(1).join(':').trim();
                }
            }
            
            const startSeconds = timeStrToSeconds(timeStr);
            parsedLines.push({ start: startSeconds, content: content });
        }
    }
    
    if (parsedLines.length === 0) {
        return null;
    }
    
    // 2. SRT 변환 로직
    for (let i = 0; i < parsedLines.length; i++) {
        const current = parsedLines[i];
        const startTime = current.start;
        
        // 종료 시간 계산
        let endTime;
        if (i < parsedLines.length - 1) {
            const nextStart = parsedLines[i + 1].start;
            // 다음 자막 시작 시간과 (현재시간 + 최대지속시간) 중 더 짧은 것을 선택
            endTime = Math.min(nextStart, startTime + maxDuration);
            
            // 예외처리: 다음 자막이 현재보다 빠르거나 같으면 2초 더함
            if (endTime <= startTime) {
                endTime = startTime + 2.0;
            }
        } else {
            // 마지막 줄 처리
            endTime = startTime + 2.0;
        }
        
        // SRT 블록 조립
        const srtBlock = `${i + 1}\n` +
            `${secondsToSrtTimestamp(startTime)} --> ${secondsToSrtTimestamp(endTime)}\n` +
            `${current.content}\n\n`;
        
        srtOutput.push(srtBlock);
    }
    
    return srtOutput.join('');
}

// [mm:ss.ms] -> seconds (float)
function timeStrToSeconds(timeStr) {
    const [minutes, seconds] = timeStr.split(':');
    return parseInt(minutes) * 60 + parseFloat(seconds);
}

// seconds (float) -> HH:MM:SS,mmm
function secondsToSrtTimestamp(secondsFloat) {
    const hours = Math.floor(secondsFloat / 3600);
    const minutes = Math.floor((secondsFloat % 3600) / 60);
    const seconds = Math.floor(secondsFloat % 60);
    const milliseconds = Math.round((secondsFloat - Math.floor(secondsFloat)) * 1000);
    
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`;
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
    alert(`Brewer Translation v1.0.0

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