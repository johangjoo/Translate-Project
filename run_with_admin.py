#!/usr/bin/env python3
"""
관리자 권한으로 Hugging Face 오디오 노이즈 제거 실행
Windows 권한 문제 해결용 스크립트
"""

import sys
import os
import ctypes
import subprocess

def is_admin():
    """현재 관리자 권한인지 확인"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """관리자 권한으로 재실행"""
    if is_admin():
        print("✅ 이미 관리자 권한으로 실행 중입니다.")
        return True
    else:
        print("🔄 관리자 권한으로 재실행합니다...")
        try:
            # 현재 스크립트를 관리자 권한으로 재실행
            script_path = os.path.join(os.path.dirname(__file__), "rnnaudio_huggingface.py")
            ctypes.windll.shell32.ShellExecuteW(
                None, 
                "runas", 
                sys.executable, 
                f'"{script_path}"', 
                None, 
                1
            )
            return True
        except Exception as e:
            print(f"❌ 관리자 권한 실행 실패: {str(e)}")
            return False

def enable_developer_mode_guide():
    """개발자 모드 활성화 가이드"""
    print("\n" + "="*60)
    print("🛠️  Windows 개발자 모드 활성화 방법 (권장)")
    print("="*60)
    print("1. Windows 설정 열기 (Win + I)")
    print("2. '업데이트 및 보안' 클릭")
    print("3. '개발자용' 메뉴 선택")
    print("4. '개발자 모드' 토글 켜기")
    print("5. 컴퓨터 재시작")
    print("\n또는 PowerShell을 관리자 권한으로 열고 다음 명령 실행:")
    print("reg add \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock\" /t REG_DWORD /f /v \"AllowDevelopmentWithoutDevLicense\" /d \"1\"")
    print("="*60)

def main():
    """메인 실행 함수"""
    print("Windows 권한 문제 해결 도구")
    print("="*40)
    
    # 현재 권한 상태 확인
    if is_admin():
        print("✅ 관리자 권한으로 실행 중")
        
        # 메인 스크립트 실행
        try:
            import rnnaudio_huggingface
            print("\n메인 프로그램을 실행합니다...")
            rnnaudio_huggingface.main()
        except Exception as e:
            print(f"❌ 실행 오류: {str(e)}")
            enable_developer_mode_guide()
    else:
        print("⚠️  일반 사용자 권한으로 실행 중")
        
        # 사용자에게 선택지 제공
        print("\n다음 중 하나를 선택하세요:")
        print("1. 관리자 권한으로 재실행 (권장)")
        print("2. 개발자 모드 활성화 가이드 보기")
        print("3. 권한 없이 실행 시도 (기본 처리 방법 사용)")
        
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            if not run_as_admin():
                print("관리자 권한 실행에 실패했습니다.")
                enable_developer_mode_guide()
        elif choice == "2":
            enable_developer_mode_guide()
        elif choice == "3":
            try:
                import rnnaudio_huggingface
                print("\n⚠️  권한 제한 모드로 실행합니다...")
                print("Hugging Face 모델 로딩이 실패할 수 있지만 기본 처리는 작동합니다.")
                rnnaudio_huggingface.main()
            except Exception as e:
                print(f"❌ 실행 오류: {str(e)}")
                enable_developer_mode_guide()
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
