from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import *
from src.database.chroma_vector_store import ChromaVectorStore
from src.crawler.cnu_food_crawler import RealTimeFoodUpdater
from .models import QuestionRequest, QuestionResponse, SourceInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_system = None
vector_store: ChromaVectorStore = None
food_updater: RealTimeFoodUpdater = None

def save_conversation(question: str, response: Dict[str, Any]):
    """대화 내용을 JSON 파일로 저장"""
    try:
        conversation_file = LOGS_DIR / "conversations.json"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": response.get("answer", ""),
            "category": response.get("category", "unknown"),
            "confidence": response.get("confidence", 0.0),
            "sources_count": len(response.get("sources", []))
        }
        
        conversations = []
        if conversation_file.exists():
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        
        conversations.append(conversation_entry)
        
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"대화 저장 실패: {e}")

def load_data_files():
    all_data = {}
    data_files = {
        "cnu_notices.json": "school_announcements",
        "graduation.json": "graduation_requirements", 
        "schedule.jsonl": "academic_calendar",
        "menu.json": "meal_guide",
        "menu_time.json": "meal_guide",
        "shuttle.json": "transportation"
    }
    
    for filename, category in data_files.items():
        try:
            file_path = DATA_DIR / "raw" / filename
            if not file_path.exists():
                continue
            
            if filename.endswith('.json'):
                encoding = 'utf-8-sig' if filename in ['menu.json', 'shuttle.json'] else 'utf-8'
                with open(file_path, 'r', encoding=encoding) as f:
                    raw_data = json.load(f)
            elif filename.endswith('.jsonl'):
                raw_data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            raw_data.append(json.loads(line))
            
            processed_data = []
            
            if filename == "shuttle.json":
                for i, (bus_name, bus_info) in enumerate(raw_data.items()):
                    title = f"{bus_name} 셔틀버스 정보"
                    content_parts = [f"노선명: {bus_name}"]
                    
                    # 운행시간 정보
                    if '첫차' in bus_info:
                        content_parts.append(f"첫차: {bus_info['첫차']}")
                    if '막차' in bus_info:
                        content_parts.append(f"막차: {bus_info['막차']}")
                    if '운행횟수' in bus_info:
                        content_parts.append(f"운행횟수: {bus_info['운행횟수']}")
                    
                    # 노선 정보
                    if '노선' in bus_info and bus_info['노선']:
                        content_parts.append("\n노선 정보:")
                        for route in bus_info['노선']:
                            content_parts.append(f"- {route}")
                    
                    # 비고 정보
                    if '비고' in bus_info and bus_info['비고']:
                        content_parts.append("\n비고:")
                        for note in bus_info['비고']:
                            content_parts.append(f"- {note}")
                    
                    # 검색 키워드 추가
                    keywords = [bus_name, "셔틀버스", "교통", "버스"]
                    if "교내" in bus_name:
                        keywords.extend(["교내순환", "캠퍼스순환", "유성캠퍼스"])
                    if "대덕" in bus_name or "보운" in bus_name:
                        keywords.extend(["대덕캠퍼스", "보운캠퍼스", "캠퍼스간"])
                    
                    content = "\n".join(content_parts)
                    content += f"\n\n검색키워드: {', '.join(keywords)}"
                    
                    processed_item = {
                        "id": f"{category}_{filename}_{i}",
                        "title": title,
                        "content": content,
                        "url": '',
                        "timestamp": ''
                    }
                    processed_data.append(processed_item)
            elif filename == "graduation.json":
                processed_item = {
                    "id": f"{category}_{filename}_0",
                    "title": raw_data.get('document_title', '졸업요건 정보_학점이수'),
                    "content": json.dumps(raw_data, ensure_ascii=False, indent=2),
                    "url": '',
                    "timestamp": raw_data.get('academic_year', '')
                }
                processed_data.append(processed_item)
            else:
                for i, item in enumerate(raw_data):
                    if isinstance(item, dict):
                        building_name = '식당'  # 기본값 설정
                        if filename == "menu_time.json":
                            building_name = item.get('name', '식당')
                        title = f"{building_name} 운영시간 안내"
                        content_parts = [f"식당: {building_name}"]
                        corners = item.get('corners', {})
                        if corners:
                            content_parts.append("\n운영시간 정보:")
                            for corner, times in corners.items():
                                if isinstance(times, list):
                                    time_str = ", ".join(times)
                                else:
                                    time_str = str(times)
                                content_parts.append(f"{corner}: {time_str}")
                        
                        # 검색을 위한 키워드 추가
                        keywords = [building_name]
                        if "제1학생회관" in building_name:
                            keywords.extend(["1학", "제1학", "1학생회관"])
                        elif "제2학생회관" in building_name:
                            keywords.extend(["2학", "제2학", "2학생회관"])
                        elif "제3학생회관" in building_name:
                            keywords.extend(["3학", "제3학", "3학생회관"])
                        
                        content = "\n".join(content_parts)
                        if keywords:
                            content += f"\n\n검색키워드: {', '.join(keywords)}"
                    elif filename == "menu.json":
                        title = f"{item.get('date', '')} {item.get('meal_type', '')} - {item.get('building', '')}"
                        content_parts = [
                            f"날짜: {item.get('date', '')}",
                            f"식사유형: {item.get('meal_type', '')}",
                            f"식당: {item.get('building', '')}",
                            f"대상: {item.get('audience', '')}"
                        ]
                        menu_items = item.get('menu_items', [])
                        if menu_items:
                            content_parts.append(f"메뉴: {', '.join(menu_items)}")
                        content = "\n".join(content_parts)
                    else:
                        title = item.get('title', item.get('subject', f"{category} 정보"))
                        content = item.get('content', item.get('text', str(item)))
                    
                        processed_item = {
                            "id": item.get('id', f"{category}_{filename}_{i}"),
                            "title": title,
                            "content": content,
                            "url": item.get('url', ''),
                            "timestamp": item.get('date', item.get('timestamp', ''))
                        }
                        processed_data.append(processed_item)
                else:
                    processed_item = {
                        "id": f"{category}_{filename}_{i}",
                        "title": f"{category} 정보",
                        "content": str(item),
                        "url": '',
                        "timestamp": ''
                    }
                processed_data.append(processed_item)
            
            if category in all_data:
                all_data[category].extend(processed_data)
                logger.info(f"{filename}: {len(processed_data)}건 추가 로드 ({category}, 총 {len(all_data[category])}건)")
            else:
                all_data[category] = processed_data
                logger.info(f"{filename}: {len(processed_data)}건 로드 ({category})")
            
        except Exception as e:
            logger.error(f"{filename} 로드 실패: {e}")
    
    return all_data

class CNUChatRAGSystem:
    
    def __init__(self, classifier_path: str, base_model_name: str, vector_store: ChromaVectorStore, 
                 chatbot_model_name: str = CHATBOT_MODEL_NAME):
        from src.core.classifier import QuestionClassifier
        
        self.classifier = QuestionClassifier(classifier_path, base_model_name)
        self.vector_store = vector_store
        self.chatbot_model_name = chatbot_model_name
        self._load_chatbot_model()
        logger.info("RAG System 초기화 완료")
    
    def _load_chatbot_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"AWQ 모델 로딩: {self.chatbot_model_name}")
            
            self.chatbot_tokenizer = AutoTokenizer.from_pretrained(self.chatbot_model_name)
            self.chatbot_model = AutoModelForCausalLM.from_pretrained(
                self.chatbot_model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.chatbot_tokenizer.pad_token is None:
                self.chatbot_tokenizer.pad_token = self.chatbot_tokenizer.eos_token
            
            logger.info("AWQ 모델 로드 완료")
                
        except Exception as e:
            logger.error(f"AWQ 모델 로드 실패: {e}")
            self.chatbot_model = self.classifier.model
            self.chatbot_tokenizer = self.classifier.tokenizer
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        try:
            category_id = self.classifier.classify(question)
            category_name = self.classifier.get_category_name(category_id)
            
            if category_id == -1:
                return {
                    "answer": "질문을 분류할 수 없습니다. 다시 질문해 주세요.",
                    "category": "unknown",
                    "confidence": 0.0,
                    "sources": []
                }
            
            relevant_docs = self.vector_store.search(
                category=category_name,
                query=question,
                top_k=5
            )
            
            if not relevant_docs or (relevant_docs and max(doc['similarity'] for doc in relevant_docs) < 0.5):
                relevant_docs = self._search_all_categories(question)
                if relevant_docs:
                    best_category = relevant_docs[0]['metadata']['category']
                    category_name = best_category
            
            answer = self._generate_rag_answer(question, relevant_docs, category_name)
            confidence = self._calculate_confidence(relevant_docs)
            
            return {
                "answer": answer,
                "category": category_name,
                "category_id": category_id,
                "confidence": confidence,
                "sources": self._format_sources(relevant_docs),
                "context_used": len(relevant_docs) > 0
            }
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return {
                "answer": "시스템 오류가 발생했습니다.",
                "category": "error",
                "confidence": 0.0,
                "sources": []
            }
    
    def _search_all_categories(self, question: str) -> List[Dict]:
        all_results = []
        all_categories = self.vector_store.get_all_categories()
        
        for category in all_categories:
            try:
                results = self.vector_store.search(
                    category=category,
                    query=question,
                    top_k=2
                )
                
                if results:
                    for result in results:
                        result['metadata']['category'] = category
                    all_results.extend(results)
                    
            except Exception as e:
                logger.error(f"카테고리 {category} 검색 실패: {e}")
        
        if all_results:
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            return all_results[:5]
        
        return []
    
    def _generate_rag_answer(self, question: str, relevant_docs: List[Dict], category: str) -> str:
        if not relevant_docs:
            return self._generate_no_context_answer(question, category)
        
        context = self._build_context(relevant_docs)
        return self._generate_qwen_answer(question, context, category)
    
    def _build_context(self, relevant_docs: List[Dict]) -> str:
        sorted_docs = sorted(relevant_docs, 
            key=lambda x: x['metadata'].get('timestamp', '0000-00-00'), 
            reverse=True)
        
        context_parts = []
        for i, doc in enumerate(sorted_docs[:5]):
            title = doc['metadata']['title']
            content = doc['content']
            doc_category = doc['metadata'].get('category', '')
            doc_date = doc['metadata'].get('timestamp', '날짜 정보 없음')
            
            if len(content) > 1200:
                content = content[:1200] + "..."
            
            category_info = f" (출처: {CATEGORY_NAMES.get(doc_category, doc_category)})" if doc_category else ""
            date_info = f" [날짜: {doc_date}]" if doc_date != '날짜 정보 없음' else ""
            context_parts.append(f"문서 {i+1}: {title}{category_info}{date_info}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_qwen_answer(self, question: str, context: str, category: str) -> str:
        """Qwen3-14B AWQ 모델 기반 전문적 답변 생성"""
        from datetime import datetime
        
        category_korean = CATEGORY_NAMES.get(category, category)
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        try:
            system_prompt = f"""충남대학교 AI 어시스턴트 (현재: {current_date})

핵심 규칙:
1. 최근/최신 질문 시: 가장 최근 날짜 문서만 선택하여 완전한 내용 제공
2. 학식 질문 시: 해당 식당 정보만 정확히 제공  
3. 한국어로 명확하고 완전한 답변 작성
4. 태그 사용 금지: <think>, <reasoning> 등
5. 답변을 끝까지 완성하여 제공

최신 문서 선택 기준: 문서 날짜를 {current_date}와 비교하여 가장 최근 것만 사용"""

            user_prompt = f"""참고 정보:
{context}

학생 질문: {question}

위 정보를 바탕으로 학생의 질문에 답변해주세요."""

            result = self._call_qwen_model(system_prompt, user_prompt)
            logger.info(f"Qwen 모델 응답: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"RAG 답변 생성 실패: {e}")
            return "시스템 오류가 발생했습니다."
    
    def _generate_no_context_answer(self, question: str, category: str) -> str:
        """정보 부족시 지능형 대안 안내 생성"""
        from datetime import datetime
        
        category_korean = CATEGORY_NAMES.get(category, category)
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        try:
            system_prompt = f"""당신은 충남대학교 전용 지능형 AI 어시스턴트입니다.
사용자가 {category_korean}에 대해 질문했으나 현재 데이터베이스에서 관련 정보를 찾을 수 없습니다.
전문적이고 체계적인 대안을 제시해주세요.

현재 날짜: {current_date}"""

            user_prompt = f"""질문: {question}

{category_korean} 관련 정보를 찾을 수 없습니다. 어떻게 도움을 받을 수 있는지 안내해주세요."""

            return self._call_qwen_model(system_prompt, user_prompt)
            
        except Exception as e:
            logger.error(f"일반 답변 생성 실패: {e}")
            return f"{category_korean} 관련 정보를 찾을 수 없습니다. 구체적인 문의는 해당 부서에 연락하시기 바랍니다."
    
    def _call_qwen_model(self, system_prompt: str, user_prompt: str) -> str:
        """Qwen3 통합 모델 인터페이스"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return self._call_transformers_model(messages)
                
        except Exception as e:
            logger.error(f"모델 호출 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def _call_transformers_model(self, messages: List[Dict[str, str]]) -> str:
        """고성능 Transformers 모델 추론 엔진"""
        try:
            import torch
            
            text = self.chatbot_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.chatbot_tokenizer(text, return_tensors="pt").to(self.chatbot_model.device)
            
            with torch.no_grad():
                outputs = self.chatbot_model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.chatbot_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    eos_token_id=self.chatbot_tokenizer.eos_token_id
                )
            
            answer = self.chatbot_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            ).strip()
            
            # <think> 태그 및 내부 추론 과정 제거
            answer = self._clean_model_response(answer)
            
            logger.info(f"Transformers 모델 생성 답변 길이: {len(answer)} 문자")
            logger.info(f"정리된 답변: {answer[:200]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"모델 호출 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def _calculate_confidence(self, relevant_docs: List[Dict]) -> float:
        """동적 신뢰도 평가 알고리즘"""
        if not relevant_docs:
            return 0.0
        
        # 최고 유사도 기반 신뢰도
        max_similarity = max(doc['similarity'] for doc in relevant_docs)
        
        # 문서 개수도 고려
        doc_count_bonus = min(len(relevant_docs) * 0.1, 0.2)
        confidence = min(max_similarity + doc_count_bonus, 1.0)
        
        return round(confidence, 2)
    
    def _format_sources(self, relevant_docs: List[Dict]) -> List[Dict]:
        """구조화된 소스 메타데이터 생성"""
        sources = []
        for doc in relevant_docs:
            metadata = doc.get('metadata', {})
            sources.append({
                "title": metadata.get('title', '제목 없음'),
                "url": metadata.get('url', ''),
                "similarity": round(doc['similarity'], 3),
                "preview": doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
            })
        return sources
    
    def _clean_model_response(self, response: str) -> str:
        """모델 응답 후처리 및 품질 검증"""
        import re
        
        # <think> 태그와 그 이후 모든 내용 제거 (닫히지 않은 태그 포함)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
        
        # 기타 추론 관련 태그들 제거
        response = re.sub(r'<reasoning>.*?</reasoning>', '', response, flags=re.DOTALL)
        response = re.sub(r'<reasoning>.*', '', response, flags=re.DOTALL)
        response = re.sub(r'<analysis>.*?</analysis>', '', response, flags=re.DOTALL)
        response = re.sub(r'<analysis>.*', '', response, flags=re.DOTALL)
        
        # 연속된 줄바꿈 정리
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        
        # 앞뒤 공백 제거
        response = response.strip()
        
        # 빈 응답이면 기본 메시지 반환
        if not response or len(response) < 10:
            return "죄송합니다. 답변을 생성하는데 문제가 발생했습니다."
        
        return response
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행될 함수"""
    logger.info("충남대 챗봇 API 서버 시작...")
    
    global rag_system, vector_store, food_updater
    
    try:
        logger.info("Chroma Vector Store 초기화 중...")
        vector_store = ChromaVectorStore(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME
        )
        
        logger.info("데이터 파일 로딩 중...")
        all_data = load_data_files()
        
        for category, documents in all_data.items():
            if documents:
                logger.info(f"{category} 벡터화 시작...")
                success = vector_store.add_documents(category, documents)
                if success:
                    logger.info(f"{category}: {len(documents)}건 벡터화 완료")
                else:
                    logger.error(f"{category} 벡터화 실패")
            else:
                logger.warning(f"{category}: 데이터 없음")
        
        logger.info("모든 데이터 벡터화 완료")
        
        logger.info("CNU Chat RAG 시스템 초기화 중...")
        rag_system = CNUChatRAGSystem(
            str(CLASSIFICATION_MODEL_PATH),
            BASE_MODEL_NAME,
            vector_store,
            chatbot_model_name=CHATBOT_MODEL_NAME
        )
        
        logger.info("실시간 학식 크롤러 시작...")
        food_updater = RealTimeFoodUpdater(vector_store)
        food_updater.start_scheduler()
        
        # 초기 메뉴 데이터 업데이트
        food_updater.update_daily_menu()
        
        logger.info("서버 초기화 완료!")
        
    except Exception as e:
        logger.error(f"서버 초기화 실패: {e}")
        raise
    
    yield
    
    # 크롤러 정리
    if food_updater:
        food_updater.stop_scheduler()
    
    logger.info("충남대 챗봇 API 서버 종료...")

app = FastAPI(
    title="CNU Chat API",
    description="충남대학교 전용 고성능 AI 챗봇 서비스 - Qwen3 RAG 시스템 기반",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 설정 (웹에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 구체적인 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/health")
async def health_check():
    global rag_system, vector_store
    
    try:
        is_healthy = (rag_system is not None and vector_store is not None)
        return {"status": "healthy" if is_healthy else "unhealthy"}
        
    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/status")
async def get_status():
    global rag_system, vector_store
    
    try:
        if not vector_store:
            return {"status": "unhealthy", "total_documents": 0, "categories": {}}
        
        categories = {}
        total_docs = 0
        
        for category in vector_store.get_all_categories():
            try:
                count = len(vector_store.search(category, "test", top_k=1000))
                categories[category] = {"document_count": count}
                total_docs += count
            except:
                categories[category] = {"document_count": 0}
        
        return {
            "status": "healthy" if rag_system else "unhealthy",
            "total_documents": total_docs,
            "categories": categories
        }
        
    except Exception as e:
        logger.error(f"상태 조회 실패: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/chat")
async def chat(request: QuestionRequest):
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    try:
        result = rag_system.answer_question(request.question)
        
        save_conversation(request.question, result)
        
        return QuestionResponse(
            answer=result["answer"],
            category=result["category"],
            category_id=result.get("category_id", -1),
            confidence=result["confidence"],
            sources=[SourceInfo(
                title=source["title"],
                url=source["url"], 
                similarity=source["similarity"],
                preview=source["preview"]
            ) for source in result["sources"]],
            context_used=result["context_used"],
            response_time=0.0
        )
        
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-menu")
async def update_menu():
    """수동으로 학식 메뉴 업데이트"""
    global food_updater
    
    if food_updater is None:
        raise HTTPException(status_code=503, detail="크롤러가 초기화되지 않았습니다.")
    
    try:
        food_updater.update_daily_menu()
        return {"status": "success", "message": "학식 메뉴 업데이트 완료"}
    except Exception as e:
        logger.error(f"메뉴 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/menu-status")
async def get_menu_status():
    """현재 크롤링된 메뉴 상태 확인"""
    try:
        menu_file = RAW_DATA_DIR / "menu_crawled.json"
        if menu_file.exists():
            with open(menu_file, 'r', encoding='utf-8') as f:
                menu_data = json.load(f)
            
            latest_date = max([item.get('date', '') for item in menu_data]) if menu_data else None
            
            return {
                "status": "healthy",
                "total_menu_days": len(menu_data),
                "latest_date": latest_date,
                "crawler_running": food_updater is not None and food_updater.is_running
            }
        else:
            return {
                "status": "no_data",
                "message": "크롤링된 메뉴 데이터가 없습니다."
            }
    except Exception as e:
        logger.error(f"메뉴 상태 확인 실패: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    # 필요한 디렉토리 생성
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"],
        log_level="info"
    )