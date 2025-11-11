@echo off
echo ========================================
echo Audio Translation Frontend 설치 스크립트
echo ========================================
echo.

:: Node.js 버전 확인
echo [1/4] Node.js 버전 확인 중...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ? Node.js가 설치되어 있지 않습니다.
    echo    https://nodejs.org 에서 Node.js를 다운로드하여 설치하세요.
    pause
    exit /b 1
)

node --version
echo ? Node.js가 설치되어 있습니다.
echo.


:: npm 의존성 설치
echo [3/4] npm 의존성 설치 중...
echo    이 과정은 몇 분이 소요될 수 있습니다...
npm install
if %errorlevel% neq 0 (
    echo ? npm 의존성 설치에 실패했습니다.
    pause
    exit /b 1
)
echo ? npm 의존성 설치가 완료되었습니다.
echo.

:: 설치 완료
echo [4/4] 설치 완료!
echo ========================================
echo ?? Audio Translation Frontend 설치 완료!
echo ========================================
echo.
echo 사용 방법:
echo   개발 모드:     npm run dev
echo   프로덕션 모드: npm start
echo   빌드:         npm run build
echo.
echo 주의사항:
echo   - FastAPI 서버가 먼저 실행되어 있어야 합니다
echo   - 서버 실행: cd .. ^&^& python run_server.py
echo.
pause
