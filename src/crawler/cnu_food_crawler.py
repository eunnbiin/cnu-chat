import requests
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from bs4 import BeautifulSoup
import threading
import schedule
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR
from src.database.chroma_vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class CNUFoodCrawler:
    def __init__(self):
        self.base_url = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_daily_menu(self, date_str: str = None) -> Dict[str, Any]:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # 파라미터 설정
            params = {
                'date': date_str,
                'lang': 'ko'
            }
            
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_menu_data(soup, date_str)
            
        except Exception as e:
            logger.error(f"메뉴 크롤링 실패 ({date_str}): {e}")
            return {}
    
    def _parse_menu_data(self, soup: BeautifulSoup, date_str: str) -> Dict[str, Any]:
        menu_data = {
            "date": date_str,
            "restaurants": {}
        }
        
        try:
            restaurants = [
                "제1학생회관", "제2학생회관", "제3학생회관", 
                "제4학생회관", "생활과학대학"
            ]
            
            for restaurant in restaurants:
                restaurant_data = self._extract_restaurant_menu(soup, restaurant, date_str)
                if restaurant_data:
                    menu_data["restaurants"][restaurant] = restaurant_data
            
            return menu_data
            
        except Exception as e:
            logger.error(f"파싱 실패: {e}")
            return {}
    
    def _extract_restaurant_menu(self, soup: BeautifulSoup, restaurant: str, date_str: str) -> Dict[str, Any]:
        restaurant_data = {
            "name": restaurant,
            "meals": {}
        }
        
        try:
            meal_types = ["조식", "중식", "석식"]
            
            for meal_type in meal_types:
                meal_data = self._extract_meal_data(soup, restaurant, meal_type)
                if meal_data:
                    restaurant_data["meals"][meal_type] = meal_data
            
            return restaurant_data if restaurant_data["meals"] else None
            
        except Exception as e:
            logger.error(f"{restaurant}  실패: {e}")
            return None
    
    def _extract_meal_data(self, soup: BeautifulSoup, restaurant: str, meal_type: str) -> Dict[str, Any]:
        try:
            # 실제 HTML 구조 -- 수정 필요
            meal_sections = soup.find_all('div', class_='menu-section')
            
            for section in meal_sections:
                if restaurant in section.get_text() and meal_type in section.get_text():
                    menu_items = []
                    price_info = {}
                    
                    items = section.find_all('span', class_='menu-item')
                    for item in items:
                        menu_text = item.get_text().strip()
                        if menu_text and menu_text not in ['', '-', '운영안함']:
                            menu_items.append(menu_text)
                    
                   
            
            return {"items": [], "available": False}
            
        except Exception as e:
            logger.error(f"{restaurant} {meal_type} 메뉴 추출 실패: {e}")
            return {"items": [], "available": False}
    
    def get_week_menu(self) -> List[Dict[str, Any]]:

        week_menus = []
        today = datetime.now()
        
        for i in range(7):
            date = today + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            daily_menu = self.get_daily_menu(date_str)
            if daily_menu:
                week_menus.append(daily_menu)
            
            time.sleep(1)  
        
        return week_menus
    
    def save_menu_data(self, menu_data: Dict[str, Any]):
        try:
            # 기존 메뉴 데이터 로드
            menu_file = RAW_DATA_DIR / "menu_crawled.json"
            
            if menu_file.exists():
                with open(menu_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 중복 제거 (같은 날짜 데이터가 있으면 업데이트)
            date_str = menu_data.get('date')
            existing_data = [item for item in existing_data if item.get('date') != date_str]
            existing_data.append(menu_data)
            
            # 날짜순 정렬
            existing_data.sort(key=lambda x: x.get('date', ''))
            
            # 파일 저장
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(menu_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"메뉴 데이터 저장 완료: {date_str}")
            return True
            
        except Exception as e:
            logger.error(f"메뉴 데이터 저장 실패: {e}")
            return False
    
    def convert_to_chatbot_format(self, menu_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chatbot_data = []
        date_str = menu_data.get('date', '')
        
        for restaurant_name, restaurant_info in menu_data.get('restaurants', {}).items():
            for meal_type, meal_info in restaurant_info.get('meals', {}).items():
                if meal_info.get('available') and meal_info.get('items'):
                    doc = {
                        "id": f"menu_{date_str}_{restaurant_name}_{meal_type}",
                        "title": f"{date_str} {meal_type} - {restaurant_name}",
                        "content": self._format_menu_content(restaurant_name, meal_type, meal_info, date_str),
                        "date": date_str,
                        "url": self.base_url
                    }
                    chatbot_data.append(doc)
        
        return chatbot_data
    
    def _format_menu_content(self, restaurant: str, meal_type: str, meal_info: Dict, date_str: str) -> str:
        """메뉴 내용을 텍스트로 포맷팅"""
        content_parts = [
            f"날짜: {date_str}",
            f"식사유형: {meal_type}",
            f"식당: {restaurant}"
        ]
        
        # 메뉴 아이템
        if meal_info.get('items'):
            menu_items = ', '.join(meal_info['items'])
            content_parts.append(f"메뉴: {menu_items}")
        
        # 가격 정보
        if meal_info.get('prices'):
            for price_type, price in meal_info['prices'].items():
                content_parts.append(f"가격({price_type}): {price}")
        
        return '\n'.join(content_parts)


class RealTimeFoodUpdater:
    """실시간 학식 정보 업데이터"""
    
    def __init__(self, vector_store: ChromaVectorStore = None):
        self.crawler = CNUFoodCrawler()
        self.vector_store = vector_store
        self.is_running = False
        
    def update_daily_menu(self):
        """일일 메뉴 업데이트"""
        try:
            logger.info("일일 메뉴 업데이트 시작")
            
            # 오늘과 내일 메뉴 크롤링
            today = datetime.now().strftime("%Y-%m-%d")
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            for date_str in [today, tomorrow]:
                menu_data = self.crawler.get_daily_menu(date_str)
                if menu_data:
                    # JSON 파일 저장
                    self.crawler.save_menu_data(menu_data)
                    
                    # 벡터 스토어 업데이트
                    if self.vector_store:
                        chatbot_data = self.crawler.convert_to_chatbot_format(menu_data)
                        if chatbot_data:
                            self.vector_store.add_documents("meal_guide", chatbot_data)
                            logger.info(f"{date_str} 메뉴 벡터 스토어 업데이트 완료")
                
                time.sleep(2)  # 서버 부하 방지
            
            logger.info("일일 메뉴 업데이트 완료")
            
        except Exception as e:
            logger.error(f"일일 메뉴 업데이트 실패: {e}")
    
    def start_scheduler(self):
        """스케줄러 시작"""
        # 매일 오전 8시, 오후 12시, 오후 6시에 업데이트
        schedule.every().day.at("08:00").do(self.update_daily_menu)
        schedule.every().day.at("12:00").do(self.update_daily_menu)
        schedule.every().day.at("18:00").do(self.update_daily_menu)
        
        self.is_running = True
        
        def run_schedule():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
        
        scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
        scheduler_thread.start()
        
        logger.info("실시간 학식 업데이트 스케줄러 시작")
    
    def stop_scheduler(self):
        """스케줄러 중지"""
        self.is_running = False
        schedule.clear()
        logger.info("실시간 학식 업데이트 스케줄러 중지")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    

    crawler = CNUFoodCrawler()
    
    today_menu = crawler.get_daily_menu()
    print("오늘 메뉴:", json.dumps(today_menu, ensure_ascii=False, indent=2))
    
    if today_menu:
        chatbot_data = crawler.convert_to_chatbot_format(today_menu)
        print(f"챗봇 데이터: {len(chatbot_data)}개 문서 생성")