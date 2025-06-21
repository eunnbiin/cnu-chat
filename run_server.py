#!/usr/bin/env python3
"""충남대 챗봇 API 서버 실행 스크립트"""

import uvicorn
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.settings import API_CONFIG, VECTOR_DB_DIR, LOGS_DIR

def main():
    """서버 실행"""
    print("🚀 충남대학교 챗봇 API 서버 시작...")
    print(f"📁 프로젝트 경로: {project_root}")
    print(f"🌐 서버 주소: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"📚 API 문서: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
    print("-" * 50)
    
    # 필요한 디렉토리 생성
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # FastAPI 서버 실행
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"],
        log_level="info"
    )

if __name__ == "__main__":
    main()