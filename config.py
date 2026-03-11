from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_model_name: str = "gpt-4.1-nano"
    query_llm_model_name: str = "gpt-4.1-nano"
    embedding_model_name: str = "text-embedding-3-large"

    # 문서 경로
    doc_folder_path: str = "./Knowledge Base"

    # 하이브리드 검색 파라미터
    # retriever_w: float = 0.5  # BM25 가중치 (1-w = Vector 가중치)
    # retriever_k: int = 5                     # 각 리트리버가 가져오는 문서 수
    # reranking_top_k: int = 5                 # 리랭킹 후 최종 문서 수
    retriever_w: float = 0.7026593644572197  # BM25 가중치 (1-w = Vector 가중치)
    retriever_k: int = 2                     # 각 리트리버가 가져오는 문서 수
    reranking_top_k: int = 8                 # 리랭킹 후 최종 문서 수

    # Cross-Encoder 리랭커
    cross_encoder_rerank_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cross_encoder_device: Optional[str] = None  # None이면 sentence-transformers가 자동 선택

    # 벡터 DB
    vector_db_path: str = "./VectorDB/game_db"
    collection_name: str = "docs"

    class Config:
        # Settings 클래스 자체의 동작 방식을 설정하는 메타 설정.
        # 위의 필드들(llm_model_name 등)이 "무슨 값을 관리하냐"라면,
        # 여기는 "어떻게 읽어오냐"를 정의한다.

        env_file = ".env"           # 오버라이드 값을 읽어올 파일 경로
        env_file_encoding = "utf-8" # .env 파일의 인코딩
        extra = "ignore"            # .env에 Settings에 없는 키(예: OPENAI_API_KEY)가 있어도
                                    # 에러 내지 않고 무시. 없으면 알 수 없는 키가 있을때 에러 발생.


settings = Settings()