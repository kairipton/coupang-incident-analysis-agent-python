llm_model_name ="gpt-5-nano"
embedding_model_name ="text-embedding-3-large"
manual_md_name = "tech_manual.md"

# Cross-Encoder 리랭커(질문, 문서) 쌍을 직접 점수화해서 재정렬하는 모델
# - 한국어/영어 혼용까지 고려해서 다국어 MS MARCO 계열을 기본값으로 둠
# - 필요하면 더 작은/빠른 모델로 바꿔도 됨
cross_encoder_rerank_model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# 필요 시 장치 지정 (예: "cpu", "cuda")
# None이면 sentence-transformers가 자동 선택
cross_encoder_device = None

vector_db_path = "./VectorDB/game_db"
collection_name = "game"