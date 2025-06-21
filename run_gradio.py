#!/usr/bin/env python3
"""충남대 챗봇 Gradio 웹 UI 실행 스크립트"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Gradio 웹 UI 실행"""
    print("🌐 충남대학교 AI 챗봇 Gradio 인터페이스 시작...")
    print("📋 사전 요구사항:")
    print("   1. API 서버가 실행 중이어야 합니다: uv run run_server.py")
    print("   2. http://localhost:8000 에서 API가 작동해야 합니다")
    print("   3. gradio 패키지가 설치되어 있어야 합니다: pip install gradio")
    print("-" * 60)
    
    # 프로젝트 경로 설정
    project_root = Path(__file__).parent
    gradio_app = project_root / "src" / "ui" / "gradio_app.py"
    
    if not gradio_app.exists():
        print(f"❌ Gradio 앱을 찾을 수 없습니다: {gradio_app}")
        return
    
    # Gradio 앱 실행
    try:
        print(f"🚀 Gradio 웹 UI 시작 중...")
        print(f"📁 앱 경로: {gradio_app}")
        print(f"🌐 웹 주소: http://localhost:7860")
        print("-" * 60)
        
        # Python 실행 명령어
        cmd = [sys.executable, str(gradio_app)]
        
        # 환경변수 설정
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        
        # 실행
        subprocess.run(cmd, cwd=str(project_root), env=env)
        
    except KeyboardInterrupt:
        print("\n🔚 Gradio 웹 UI 종료")
    except Exception as e:
        print(f"❌ Gradio 웹 UI 실행 실패: {e}")
        print("\n💡 해결 방법:")
        print("1. gradio 설치: pip install gradio")
        print("2. API 서버 실행: uv run run_server.py")
        print("3. 포트 7860이 사용 가능한지 확인")

if __name__ == "__main__":
    main()