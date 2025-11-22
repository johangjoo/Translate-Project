@echo off
chcp 65001 >nul
echo ========================================
echo Brewer Translation Frontend 시작
echo ========================================
echo.

:: 의존성 확인
if not exist "node_modules" (
    echo ❌ node_modules 폴더가 없습니다.
    echo    먼저 install.bat을 실행하여 의존성을 설치하세요.
    pause
    exit /b 1
)

:: 백엔드 서버 상태 확인 및 자동 시작
echo 백엔드 서버 상태 확인 중...
curl -s http://localhost:8000/api/v1/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  백엔드 서버가 실행되지 않았습니다. 자동으로 시작합니다...
    start cmd /k "cd .. && python run_server.py"
    echo 서버가 시작될 때까지 잠시 기다립니다...
    timeout /t 10 /nobreak >nul
    
    :: 서버 시작 후 다시 확인
    curl -s http://localhost:8000/api/v1/health >nul 2>&1
    if %errorlevel% neq 0 (
        echo ⚠️  서버 시작 중... 잠시 후 앱이 시작됩니다.
    ) else (
        echo ✅ 백엔드 서버가 시작되었습니다.
    )
) else (
    echo ✅ 백엔드 서버가 실행 중입니다.
)
echo.

:: Electron 앱 시작
echo 🚀 Electron 앱을 시작합니다...
npm start

echo.
echo 앱이 종료되었습니다.
pause
