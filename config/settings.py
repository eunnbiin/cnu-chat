from pathlib import Path
from typing import Dict, Any

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# 모델 설정
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "fine-tuned-qwen3-0_6b"
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"

# 카테고리 정의
CATEGORIES = {
    0: "graduation_requirements",
    1: "school_announcements", 
    2: "academic_calendar",
    3: "meal_guide",
    4: "transportation"
}

CATEGORY_NAMES = {
    "graduation_requirements": "졸업 요건",
    "school_announcements": "학교 공지사항",
    "academic_calendar": "학사 일정", 
    "meal_guide": "학식 식단",
    "transportation": "교통/셔틀버스"
}

# 모델 설정
CHATBOT_MODEL_NAME = "Qwen/Qwen3-8B-AWQ"  # 답변 생성용 14B AWQ 양자화 모델
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # 임베딩 모델
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"  # 리랭커 모델

# Vector DB 설정 (ChromaDB + Qwen3 임베딩)
VECTOR_DB_CONFIG = {
    "provider": "chromadb",
    "persist_directory": str(VECTOR_DB_DIR),
    "embedding_model": EMBEDDING_MODEL_NAME,
    "reranker_model": RERANKER_MODEL_NAME,
    "chunk_size": 500,
    "chunk_overlap": 50
}

# RAG 설정
RAG_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "max_context_length": 2000,
    "temperature": 0.1,
    "max_new_tokens": 500
}

# API 설정
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True
}

# 로깅 설정
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "chatbot.log"),
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}