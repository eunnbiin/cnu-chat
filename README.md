# CNU 챗봇

충남대학교 학사정보 질답 시스템입니다.

## 기능

졸업요건, 학사일정, 학식, 셔틀버스 등 학교 정보를 검색하고 답변합니다.

## 설치 및 실행

```bash
pip install -r requirements.txt
bash chatbot.sh
```

- 웹 UI: http://localhost:7860
- API: http://localhost:8000

## 개별 실행

```bash
python run_server.py  # API 서버
python run_gradio.py  # 웹 UI
```

## API 사용법

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "졸업학점이 몇 학점인가요?"}'
```

## 모델 설정

`config/settings.py`에서 모델 변경 가능:

```python
CHATBOT_MODEL_NAME = "Qwen/Qwen3-8B-AWQ"
```

## 요구사항

- Python 3.8+
- CUDA GPU (16GB+ VRAM 권장)