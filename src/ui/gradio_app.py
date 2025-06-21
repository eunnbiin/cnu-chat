import gradio as gr
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

CATEGORY_NAMES = {
    'graduation_requirements': '졸업 요건',
    'school_announcements': '학교 공지사항',
    'academic_calendar': '학사 일정',
    'meal_guide': '학식 식단',
    'transportation': '교통/셔틀버스',
    'unknown': '분류 불가',
    'error': '오류'
}

def check_api_status() -> bool:
    """API 서버 상태 확인"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API 상태 확인 실패: {e}")
        return False

def get_system_status() -> Dict[str, Any]:
    """시스템 상태 정보 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"시스템 상태 조회 실패: {e}")
    return {}

def send_question_to_api(question: str) -> Dict[str, Any]:
    """질문을 API에 전송하고 답변 받기"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=120  # 답변 생성 시간을 고려한 더 긴 타임아웃
        )
        
        if response.status_code == 200:
            json_response = response.json()
            logger.info(f"API 성공 응답 받음: {json_response.get('answer', 'No answer')[:50]}...")
            return json_response
        else:
            error_response = {
                "error": f"API 오류 (상태코드: {response.status_code})",
                "detail": response.text
            }
            logger.error(f"API 오류: {error_response}")
            return error_response
    except requests.exceptions.Timeout:
        return {"error": "요청 시간 초과", "detail": "서버 응답이 너무 오래 걸립니다."}
    except Exception as e:
        return {"error": "연결 오류", "detail": str(e)}

def format_response_with_metadata(response: Dict[str, Any]) -> str:
    """응답을 메타데이터와 함께 포맷팅"""
    logger.info(f"포맷팅 시작 - 응답 키들: {list(response.keys())}")
    
    if "error" in response:
        error_msg = f" **오류 발생**: {response['error']}\n\n상세: {response.get('detail', '')}"
        logger.info(f"에러 응답 포맷팅: {error_msg}")
        return error_msg
    
    # 기본 답변
    answer = response.get("answer", "답변을 생성할 수 없습니다.")
    logger.info(f"원본 답변 길이: {len(answer)} 문자")
    
    # 답변이 너무 길면 자르기 (Gradio 렌더링 문제 방지)
    if len(answer) > 3500:
        answer = answer[:3500] + "\n\n...(답변이 길어 일부만 표시됩니다)"
        logger.info("답변이 너무 길어서 자름")
    
    # 최종 응답 구성 (메타데이터 제거)
    formatted_response = answer
    
    logger.info(f"최종 포맷된 응답 길이: {len(formatted_response)} 문자")
    logger.info(f"최종 포맷된 응답: {formatted_response[:200]}...")
    
    return formatted_response

def chat_stream_function(message: str, history: List[Dict[str, str]]):
    """스트리밍 채팅 함수 - 사용자 메시지를 즉시 표시하고 답변을 단계별로 생성"""
    if not message.strip():
        return "", history
    
    # 1. 사용자 메시지를 즉시 히스토리에 추가하고 yield
    history.append({"role": "user", "content": message})
    logger.info(f"사용자 메시지 즉시 표시: {message}")
    yield "", history
    
    # 2. "답변 생성 중..." 메시지 표시
    thinking_msg = "..."
    history.append({"role": "assistant", "content": thinking_msg})
    yield "", history
    
    # API 상태 확인
    if not check_api_status():
        error_msg = "API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.\n\n실행 명령: `uv run run_server.py`"
        history[-1] = {"role": "assistant", "content": error_msg}
        yield "", history
        return
    
    # 3. API 호출 및 응답 생성
    try:
        start_time = time.time()
        logger.info(f"API 호출 시작: {message}")
        response = send_question_to_api(message)
        total_time = time.time() - start_time
        logger.info(f"API 응답 받음: 응답시간 {total_time:.2f}초")
        
        # 응답 포맷팅
        formatted_response = format_response_with_metadata(response)
        logger.info(f"포맷된 응답: {formatted_response[:100]}...")
        
        # 4. 실제 답변으로 교체
        history[-1] = {"role": "assistant", "content": formatted_response}
        yield "", history
        
    except Exception as e:
        error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        history[-1] = {"role": "assistant", "content": error_msg}
        logger.error(f"채팅 함수 오류: {e}")
        yield "", history

def chat_function(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """동기 버전 채팅 함수 (호환성 유지)"""
    # 스트리밍 함수의 마지막 결과 반환
    result = None
    for result in chat_stream_function(message, history):
        continue
    return result if result else ("", history)

def get_system_info() -> str:
    """시스템 정보 텍스트 생성"""
    if not check_api_status():
        return " **API 서버 연결 실패**\n\n서버를 실행해주세요"
    
    system_status = get_system_status()
    if not system_status:
        return " **시스템 상태 정보를 가져올 수 없습니다**"
    
    info_text = "**API 서버**: 연결됨\n\n"
    info_text += f"**상태**: {system_status.get('status', 'Unknown')}\n"
    info_text += f"**총 문서**: {system_status.get('total_documents', 0)}개\n\n"
    
    categories = system_status.get('categories', {})
    if categories:
        info_text += "**카테고리별 문서**:\n"
        for category, info in categories.items():
            count = info.get('document_count', 0) if isinstance(info, dict) else 0
            name = CATEGORY_NAMES.get(category, category)
            info_text += f"• {name}: **{count}**개\n"
    else:
        info_text += "**문서 정보**: 로딩 중...\n"
    
    return info_text

def create_example_inputs() -> List[str]:
    """예시 질문들"""
    return [
        "국가장학금 신청 방법은?",
        "중간고사 일정은 언제야?",
        "오늘 학식 메뉴는?",
        "셔틀버스 시간표 알려줘",
        "최신 공지사항 있어?",
        "졸업 학점은 몇 학점이야?"
    ]

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # CSS 스타일
    css = """
    .gradio-container {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    """
    
    # 메인 인터페이스
    with gr.Blocks(
        title="충남대학교 AI 챗봇",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        
        # 헤더
        gr.Markdown(
            """
            # 충남대학교 AI 챗봇
            """
        )
        
        with gr.Row():
            # 메인 채팅 영역
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "안녕하세요! 충남대학교 AI 챗봇입니다. 궁금한 것이 있으시면 언제든 물어보세요!"}],
                    height=600,
                    show_label=False,
                    type="messages"
                )
                
                msg = gr.Textbox(
                    placeholder="궁금한 것을 물어보세요...",
                    show_label=False,
                    scale=4
                )
                
                with gr.Row():
                    submit_btn = gr.Button("전송", variant="primary", scale=1)
                    clear_btn = gr.Button("대화 지우기", scale=1)
            
            # 사이드바
            with gr.Column(scale=1):
                gr.Markdown("### 시스템 정보")
                
                system_info = gr.Markdown(
                    get_system_info(),
                    every=30  # 30초마다 자동 업데이트
                )
                
                refresh_btn = gr.Button("상태 새로고침", size="sm")
                
        
        # 예시 질문들
        gr.Markdown("### 예시 질문")
        example_inputs = create_example_inputs()
        examples = gr.Examples(
            examples=example_inputs,
            inputs=msg,
            label=None
        )
        
        # 이벤트 핸들러
        def respond_stream(message, chat_history):
            """스트리밍 응답 함수"""
            for result in chat_stream_function(message, chat_history):
                yield result
        
        def clear_chat():
            return [{"role": "assistant", "content": "안녕하세요! 충남대학교 AI 챗봇입니다. 졸업요건 / 공지사항 / 학사일정 / 학식 / 셔틀버스 등에 대해 궁금한 것이 있으면 언제든 물어보세요"}]
        
        def refresh_system_info():
            return get_system_info()
        
        # 이벤트 연결 (스트리밍 사용)
        msg.submit(respond_stream, [msg, chatbot], [msg, chatbot])
        submit_btn.click(respond_stream, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_chat, None, chatbot)
        refresh_btn.click(refresh_system_info, None, system_info)
        
    
    return demo

def main():
    """메인 실행 함수"""
    logger.info("충남대학교 AI 챗봇 Gradio 인터페이스 시작...")
    
    # Gradio 인터페이스 생성
    demo = create_gradio_interface()
    
    # 서버 실행
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True,
        quiet=False,
        favicon_path=None,
        ssl_verify=False
    )

if __name__ == "__main__":
    main()