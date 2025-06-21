import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class QuestionClassifier:
    """질문 분류를 위한 클래스"""
    
    def __init__(self, model_path: str, base_model_name: str):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """모델과 토크나이저 로드"""
        try:
            logger.info(f"Loading classification model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager"
            )
            
            # LoRA 어댑터 로드
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            logger.info("Classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load classification model: {e}")
            raise
    
    def classify(self, question: str) -> int:
        """질문을 분류하여 카테고리 번호 반환"""
        try:
            # 프롬프트 구성
            prompt = """
Classify the question into one of these categories:

- Graduation Requirements (0): 졸업요건, 학점, 전공, 이수, 학위, 졸업논문
- School Announcements (1): 공지사항, 알림, 안내, 소식, 뉴스, 발표
- Academic Calendar (2): 학사일정, 개강, 종강, 시험, 휴학, 복학, 수강신청
- Meal Guide (3): 학식, 식단, 메뉴, 식당, 운영시간, 1학, 2학, 3학, 학생회관, 급식
- Commuting/Shuttle Bus (4): 셔틀버스, 교통, 버스, 통학, 시간표, 노선

Examples:
- "1학 운영시간" → Meal Guide (3)
- "제1학생회관 식당" → Meal Guide (3)  
- "학식 메뉴" → Meal Guide (3)

Return only the number in JSON format: { "answer": 3 }
"""
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ]
            
            # 토큰화
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 응답 디코딩
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            ).strip()
            
            # 답변 추출
            category = self._extract_category(response)
            logger.info(f"Question classified as category: {category}")
            
            return category
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return -1  # 분류 실패 시 기본값
    
    def _extract_category(self, response: str) -> int:
        """응답에서 카테고리 번호 추출"""
        try:
            # JSON 형태로 답변이 있는지 확인
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                answer = data.get("answer", -1)
                if isinstance(answer, int) and 0 <= answer <= 4:
                    return answer
            
            # JSON이 없으면 숫자 직접 추출
            for char in response:
                if char.isdigit():
                    num = int(char)
                    if 0 <= num <= 4:
                        return num
                        
        except Exception as e:
            logger.warning(f"Failed to extract category from response: {response}, error: {e}")
        
        return -1  # 추출 실패
    
    def get_category_name(self, category_id: int) -> str:
        category_mapping = {
            0: "graduation_requirements",
            1: "school_announcements", 
            2: "academic_calendar",
            3: "meal_guide",
            4: "transportation"
        }
        return category_mapping.get(category_id, "unknown")