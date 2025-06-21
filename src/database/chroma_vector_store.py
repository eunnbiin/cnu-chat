import numpy as np
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import chromadb
from chromadb.config import Settings
import re

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Chroma DB 기반 Vector Store with Qwen3 Embedding and Reranker"""
    
    def __init__(self, persist_directory: str, embedding_model_name: str, reranker_model_name: str):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        
        # Chroma 클라이언트 초기화
        self._init_chroma_client()
        
        # 모델 로드
        self._load_models()
        
        # 한국어 전처리를 위한 설정
        self.korean_stopwords = {
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', 
            '와', '과', '도', '만', '에게', '한테', '께', '부터', '까지', 
            '의', '에다', '라서', '여서', '고', '며', '면서', '지만', 
            '하지만', '그런데', '그리고', '또한', '그래서'
        }
        
        logger.info(f"Chroma Vector Store 초기화: {self.persist_directory}")
    
    def _init_chroma_client(self):
        """Chroma 클라이언트 초기화"""
        try:
            # Chroma 설정
            settings = Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False
            )
            
            # Persistent 클라이언트 생성
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=settings
            )
            
            logger.info("✅ Chroma 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ Chroma 클라이언트 초기화 실패: {e}")
            raise
    
    def _load_models(self):
        """임베딩 모델과 리랭커 모델 로드"""
        try:
            logger.info(f"임베딩 모델 로딩: {self.embedding_model_name}")
            
            # 임베딩 모델 로드
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, padding_side='left')
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )
            self.embedding_model.eval()
            self.max_embedding_length = 8192
            
            logger.info(f"리랭커 모델 로딩: {self.reranker_model_name}")
            
            # 리랭커 모델 로드
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name, padding_side='left')
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                self.reranker_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
            self.reranker_model.eval()
            
            # 리랭커 특수 토큰 설정
            self.token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
            self.max_rerank_length = 8192
            
            # 리랭커 프롬프트 템플릿
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
            self.suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
            
            logger.info("✅ 임베딩 및 리랭커 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def _preprocess_korean_text(self, text: str) -> str:
        """한국어 텍스트 전처리"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 마크다운 문법 제거
        text = re.sub(r'[#*`]+', '', text)
        # 특수문자 일부 제거 (의미있는 것은 보존)
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """마지막 토큰 풀링"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """쿼리용 상세 지시사항 생성"""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def _get_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """텍스트의 임베딩 벡터 생성"""
        # 텍스트 전처리
        processed_text = self._preprocess_korean_text(text)
        
        # 쿼리인 경우 instruction 추가
        if is_query:
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            input_text = self.get_detailed_instruct(task, processed_text)
        else:
            # 문서는 그대로 사용
            input_text = processed_text
        
        # 토크나이징
        batch_dict = self.embedding_tokenizer(
            [input_text],
            padding=True,
            truncation=True,
            max_length=self.max_embedding_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.embedding_model.device) for k, v in batch_dict.items()}
        
        # 임베딩 생성
        with torch.no_grad():
            outputs = self.embedding_model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # 정규화
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().flatten()
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """쿼리의 임베딩 벡터 생성 (instruction 포함)"""
        return self._get_embedding(query, is_query=True)
    
    def _format_rerank_instruction(self, query: str, doc: str) -> str:
        """리랭커 입력 형식 포맷팅"""
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
    
    def _process_rerank_inputs(self, pairs: List[str]) -> Dict:
        """리랭커 입력 토크나이징 및 처리"""
        inputs = self.reranker_tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_rerank_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # 프롬프트 토큰 추가
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        inputs = self.reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_rerank_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.reranker_model.device)
        return inputs
    
    @torch.no_grad()
    def _compute_rerank_scores(self, inputs: Dict) -> List[float]:
        batch_logits = self.reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_logits[:, self.token_true_id]
        false_vector = batch_logits[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return documents
        
        if not hasattr(self, 'reranker_model') or self.reranker_model is None:
            logger.info(f"리랭커 모델 없음: {len(documents)}개 문서 순서 유지")
            return documents
            
        try:
            pairs = []
            for doc in documents:
                content = doc['content'][:500]  # 길이 제한
                formatted_pair = self._format_rerank_instruction(query, content)
                pairs.append(formatted_pair)
            
            # 입력 처리 및 점수 계산
            inputs = self._process_rerank_inputs(pairs)
            rerank_scores = self._compute_rerank_scores(inputs)
            
            # 문서에 리랭커 점수 추가
            for i, doc in enumerate(documents):
                doc['rerank_score'] = rerank_scores[i]
                # 원래 유사도와 리랭커 점수를 결합 (가중평균)
                original_sim = doc.get('similarity', 0.0)
                doc['final_score'] = 0.3 * original_sim + 0.7 * rerank_scores[i]
            
            # 최종 점수로 재정렬
            documents.sort(key=lambda x: x['final_score'], reverse=True)
            
            # final_score를 similarity로 업데이트
            for doc in documents:
                doc['similarity'] = doc['final_score']
                del doc['final_score']
                del doc['rerank_score']
            
            logger.info(f"리랭커로 {len(documents)}개 문서 재순위화 완료")
            return documents
            
        except Exception as e:
            logger.error(f"리랭킹 실패: {e}")
            return documents  # 실패 시 원본 반환
    
    def create_collection(self, category: str) -> bool:
        """카테고리별 컬렉션 생성"""
        try:
            collection_name = f"{category}_collection"
            
            # 기존 컬렉션 확인
            try:
                self.chroma_client.get_collection(collection_name)
                logger.info(f"기존 컬렉션 로드: {collection_name}")
            except Exception:
                # 새 컬렉션 생성
                self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"category": category}
                )
                logger.info(f"새 컬렉션 생성: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 생성 실패 ({category}): {e}")
            return False
    
    def add_documents(self, category: str, documents: List[Dict[str, Any]]) -> bool:
        """문서들을 Chroma DB에 추가"""
        try:
            collection_name = f"{category}_collection"
            
            # 컬렉션 생성 또는 가져오기
            if not self.create_collection(category):
                return False
            
            collection = self.chroma_client.get_collection(collection_name)
            
            # 기존 문서 수 확인
            existing_count = collection.count()
            if existing_count >= len(documents):
                logger.info(f"{collection_name}에 이미 {existing_count}개 문서가 있습니다. 추가를 건너뜁니다.")
                return True
            
            # 문서 전처리 및 임베딩 생성
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []
            
            for doc in documents:
                doc_id = doc.get('id', f"doc_{uuid.uuid4()}")
                
                metadata = {
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'timestamp': doc.get('timestamp', ''),
                    'category': category
                }
                
                # 제목과 내용을 결합하여 임베딩 생성
                text_for_embedding = f"{metadata['title']} {doc.get('content', '')}"
                embedding = self._get_embedding(text_for_embedding)
                
                ids.append(doc_id)
                embeddings.append(embedding.tolist())
                metadatas.append(metadata)
                documents_text.append(doc.get('content', ''))
            
            # Chroma에 문서 추가
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            logger.info(f"{collection_name}에 {len(documents)}개 문서 추가 완료")
            return True
            
        except Exception as e:
            logger.error(f"문서 추가 실패 ({category}): {e}")
            return False
    
    def search(self, category: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Chroma DB에서 검색 + 리랭킹"""
        try:
            collection_name = f"{category}_collection"
            
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                logger.warning(f"컬렉션을 찾을 수 없음: {collection_name}")
                return []
            
            if collection.count() == 0:
                logger.warning(f"컬렉션이 비어있음: {collection_name}")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self._get_query_embedding(query)
            
            # Chroma에서 검색 (리랭킹을 위해 더 많이 가져옴)
            search_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 2, collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            # 결과 포맷팅
            results = []
            if search_results['documents'] and search_results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    search_results['documents'][0],
                    search_results['metadatas'][0],
                    search_results['distances'][0]
                )):
                    # 거리를 유사도로 변환 (Chroma는 L2 거리 사용)
                    similarity = 1.0 / (1.0 + distance)
                    
                    # 임계값 필터링
                    if similarity >= 0.3:
                        results.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'distance': distance
                        })
            
            # 리랭커로 재순위화
            if results:
                results = self._rerank_documents(query, results)
            
            # 최종 top_k개만 반환
            final_results = results[:top_k]
            
            logger.info(f"{collection_name}에서 {len(final_results)}개 문서 찾음 (Chroma + 리랭킹)")
            return final_results
            
        except Exception as e:
            logger.error(f"검색 실패 ({category}): {e}")
            return []
    
    def get_collection_info(self, category: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            collection_name = f"{category}_collection"
            
            try:
                collection = self.chroma_client.get_collection(collection_name)
                document_count = collection.count()
            except Exception:
                return {"error": f"컬렉션을 찾을 수 없음: {collection_name}"}
            
            return {
                "category": category,
                "document_count": document_count,
                "collection_name": collection_name,
                "embedding_model": self.embedding_model_name,
                "reranker_model": self.reranker_model_name,
                "vector_db": "ChromaDB"
            }
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패 ({category}): {e}")
            return {"error": str(e)}
    
    def get_all_categories(self) -> List[str]:
        """모든 카테고리 목록 반환"""
        try:
            collections = self.chroma_client.list_collections()
            categories = []
            
            for collection in collections:
                # collection.name에서 '_collection' 제거하여 카테고리 추출
                if collection.name.endswith('_collection'):
                    category = collection.name.replace('_collection', '')
                    categories.append(category)
            
            return categories
            
        except Exception as e:
            logger.error(f"카테고리 목록 조회 실패: {e}")
            return []
    
    def delete_collection(self, category: str) -> bool:
        """컬렉션 삭제"""
        try:
            collection_name = f"{category}_collection"
            
            try:
                self.chroma_client.delete_collection(collection_name)
                logger.info(f"컬렉션 삭제: {collection_name}")
                return True
            except Exception:
                logger.warning(f"컬렉션을 찾을 수 없음: {collection_name}")
                return False
            
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패 ({category}): {e}")
            return False