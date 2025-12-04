"""
Audio Translation API 서버 실행 스크립트 (통합 버전)
- Whisper STT + Qwen3-14b 번역
- 하나의 서버로 모든 기능!

사용법:
    python run_server.py

주의:
    - 이 파일은 Translate-Project 폴더 (루트)에 위치해야 합니다
    - qwen3-14b-lora-10ratio 폴더가 있어야 합니다
"""

import uvicorn
import sys
import os
from pathlib import Path

def check_environment():
    """환경 체크"""
    print("\n" + "="*70)
    print("  🔍 환경 체크 중...")
    print("="*70 + "\n")
    
    # 1. 프로젝트 루트 확인
    PROJECT_ROOT = Path(__file__).resolve().parent
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    
    # 2. api 폴더 확인
    api_folder = PROJECT_ROOT / "api"
    if not api_folder.exists():
        print(f"\n오류: api 폴더를 찾을 수 없습니다!")
        print(f"   예상 위치: {api_folder}")
        return False
    
    print(f"api 폴더 확인: {api_folder}")
    
    # 3. 필수 파일 확인
    required_files = [
        "api/__init__.py",
        "api/main.py",
        "api/routes.py",
        "api/inference.py",
        "api/translation/__init__.py",
        "api/translation/factory.py"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = PROJECT_ROOT / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"\n❌ 오류: 다음 파일들이 없습니다:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # 4. 번역 모델 경로 확인 (14B 모델)
    print()
    MODEL_PATH = PROJECT_ROOT / "qwen3-14b-lora-10ratio"
    
    if not MODEL_PATH.exists():
        print(f"⚠️  경고: 번역 모델을 찾을 수 없습니다!")
        print(f"   예상 경로: {MODEL_PATH}")
        print()
        print(f"   다음을 확인하세요:")
        print(f"   1. 모델 폴더 이름이 'qwen3-14b-lora-10ratio'인지")
        print(f"   2. 모델 폴더가 프로젝트 루트에 있는지")
        print(f"   3. api/config.py의 TRANSLATION_BASE_MODEL 경로가 올바른지")
        print()
        print(f"   ⚠️  번역 기능이 작동하지 않을 수 있습니다!")
        print(f"   계속 진행하시겠습니까? (Y/n): ", end="")
        
        response = input().strip().lower()
        if response == 'n':
            print("\n서버 시작을 취소합니다.")
            return False
    else:
        print(f"✅ 번역 모델 확인: {MODEL_PATH}")
    
    # 5. 필수 패키지 확인
    print()
    print("📦 필수 패키지 확인 중...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "torch",
        "whisper",
        "transformers",
        "peft"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  경고: 다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print(f"   설치 명령어:")
        print(f"   pip install {' '.join(missing_packages)}")
        print()
        print(f"   계속 진행하시겠습니까? (Y/n): ", end="")
        
        response = input().strip().lower()
        if response == 'n':
            print("\n서버 시작을 취소합니다.")
            return False
    
    print()
    print("="*70)
    print("✅ 모든 환경 체크 완료!")
    print("="*70)
    
    return True


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("  🎤 → 📝 → 🌐 Audio Translation API Server")
    print("  Whisper STT + Qwen3-14b Translation")
    print("="*70)
    
    # 환경 체크
    if not check_environment():
        sys.exit(1)
    
    # 서버 실행
    print("\n🚀 서버 시작 중...\n")
    
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # 모델 로딩 후엔 reload 끄기
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n서버를 종료합니다...")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()