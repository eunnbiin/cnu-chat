# CNU 챗봇

충남대학교 학사정보 질답 시스템입니다. RAG 기반으로 졸업요건, 학사일정, 학식 메뉴, 교통정보 등에 대한 질문에 답변합니다.

## 주요 기능

- 졸업요건 및 이수학점 조회
- 학사일정 및 중요 공지사항 확인
- 학식 메뉴 및 운영시간 안내
- 셔틀버스 노선 및 시간표 정보
- 실시간 크롤링을 통한 최신 정보 제공

## 시스템 구조

- **메인 모델**: Qwen3-8B-AWQ (답변 생성)
- **분류 모델**: fine-tuned-qwen3-0.6b (질문 분류)
- **임베딩 모델**: Qwen3-Embedding-0.6B (벡터 검색)
- **리랭커**: Qwen3-Reranker-0.6B (문서 재순위화)
- **벡터 DB**: ChromaDB (지식베이스 저장)

## 설치 및 실행

### 빠른 시작

```bash
pip install -r requirements.txt
bash chatbot.sh
```

실행 후 접속:
- 웹 UI: http://localhost:7860 (Gradio 인터페이스)
- API 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs

### 개별 실행

```bash
# API 서버만 실행
python run_server.py

# 웹 UI만 실행
python run_gradio.py
```

## API 사용법

### 챗봇 대화

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "졸업학점이 몇 학점인가요?"}'
```

### 서버 상태 확인

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

## 데이터

- `data/train.json` - 질문 분류 학습 데이터
- `data/raw/graduation.json` - 졸업요건 정보
- `data/raw/menu.json` - 학식 메뉴 정보
- `data/raw/shuttle.json` - 셔틀버스 정보
- `data/vector_db/` - ChromaDB 벡터 저장소

## 설정

### 모델 변경

`config/settings.py`에서 사용할 모델을 변경할 수 있습니다:

```python
CHATBOT_MODEL_NAME = "Qwen/Qwen3-8B-AWQ"  # 메인 답변 생성 모델
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # 임베딩 모델
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"  # 리랭킹 모델
```

### 카테고리 설정

질문 분류를 위한 카테고리는 `CATEGORY_MAPPING`에서 수정할 수 있습니다.

## 로그 및 디버깅

- `server.log` - API 서버 실행 로그
- `gradio.log` - Gradio UI 실행 로그
- `outputs/realtime_output.json` - 자동 테스트 결과

## 요구사항

### 하드웨어
- CUDA 지원 GPU (최소 16GB VRAM 권장)
- 시스템 메모리 16GB 이상

### 소프트웨어
- Python 3.8 이상
- CUDA 11.8 이상
- PyTorch 2.0 이상

## 주의사항

- 모델 로딩 시 인터넷 연결이 필요합니다
- 처음 실행시 모델 다운로드로 시간이 걸릴 수 있습니다
- GPU 메모리 부족시 모델 크기를 줄여서 사용하세요