# 쿠팡 개인정보 유출 사고 분석 AI 에이전트

> ⚠️ **이 프로젝트는 개인 포트폴리오입니다. 쿠팡과 무관한 개인 학습 프로젝트입니다.**

> 쿠팡 유출 사고 관련 질문에 AI가 실시간으로 답변하는 RAG 에이전트.  
> 단순 Vector 검색을 넘어 **하이브리드 검색 + CrossEncoder 리랭킹 + RAGAS 품질 평가**까지 구현한 포트폴리오 프로젝트입니다.

**[▶ 데모 바로가기](http://157.173.102.95:8101/)**

---

## 프로젝트 소개

사용자가 쿠팡 유출 사고에 관한 질문을 입력하면, AI 에이전트가 내부 지식 베이스를 검색하고 필요 시 웹 검색 도구를 활용하여 정확한 답변을 생성합니다.  
단순 RAG를 넘어 **Multi-Query 생성 → 하이브리드 검색 → CrossEncoder 리랭킹 → Tool Calling → RAGAS 품질 평가** 까지 이어지는 파이프라인을 구현했습니다.

---

## 시스템 구성

```
Unity Client
        ↕ HTTP (REST API)
FastAPI Server (Python)
        ↕
LangGraph Agent
  ├── DSPy: Multi-Query 생성
  ├── Hybrid Search: BM25 + Vector (ChromaDB)
  ├── CrossEncoder: 리랭킹
  ├── Tool Calling: 웹 검색 (Tavily)
  └── RAGAS: 응답 품질 평가
```

---

## AI 파이프라인 흐름

```
START
  → question       : 사용자 질문 수신 및 저장
  → multi_query    : DSPy로 검색 쿼리 3~5개 자동 생성
  → hybrid_search  : BM25 + Vector 하이브리드 검색 → CrossEncoder 리랭킹
  → tool_call      : LLM이 도구 사용 여부 판단
  → [tools]        : 필요 시 웹 검색 실행 후 tool_call로 재진입
  → final_answer   : 최종 답변 스트리밍 생성
  → summary        : DSPy로 대화 요약 갱신
  → evaluate       : RAGAS 품질 평가 (Faithfulness, AnswerRelevancy, ContextPrecision)
  → graph_end
END
```

---

## 주요 기술 스택

| 분류 | 기술 |
|---|---|
| 서버 | FastAPI, Uvicorn |
| AI 오케스트레이션 | LangGraph |
| LLM | OpenAI GPT-4.1-nano |
| 임베딩 | text-embedding-3-large |
| 프롬프트 최적화 | DSPy (Multi-Query 생성, 대화 요약) |
| 검색 | BM25 (rank-bm25) + ChromaDB 벡터 검색 (하이브리드) |
| 리랭킹 | CrossEncoder (mmarco-mMiniLMv2-L12-H384-v1) |
| 외부 도구 | Tavily 웹 검색 |
| RAG 평가 | RAGAS (Faithfulness / AnswerRelevancy / ContextPrecision) |
| 하이퍼파라미터 튜닝 | Optuna (Bayesian Optimization) |
| 모니터링 | LangSmith, Portainer |
| 클라이언트 | Unity |
| 배포 | Docker |

---

## 핵심 구현 포인트 및 기술 채택 이유

### 1. LangGraph 상태 기반 에이전트
- LangChain은 선형 연결 구조라 유연성이 떨어지고 디버깅이 어렵다는 단점이 있어 LangGraph를 채택
- `TypedDict` State로 노드 간 데이터 흐름 명시적 관리
- uid별 `MemorySaver`로 사용자별 멀티턴 대화 기억 유지
- `MemorySaver` 인스턴스를 클래스 변수로 캐싱하여 세션 간 메모리 보존

### 2. Hybrid Search + Reranking
- 단순 Vector DB 유사도 검색의 한계를 넘기 위해 BM25와 함께 사용하는 Hybrid Search 채택
- **BM25**(키워드 정밀도)와 **Vector Search**(의미 유사도)를 RRF(Reciprocal Rank Fusion)로 결합
- Hybrid Search 결과는 질문과의 유사도 순으로 정렬된 상태가 아니므로, CrossEncoder로 (질문, 문서) 쌍을 직접 점수화하여 최상위 n개만 선별

### 3. DSPy 기반 프롬프트 최적화
- 원래 모든 프롬프트를 DSPy로 자동화하는 것이 목표였으나, 자유형 질답 중심의 프로젝트 특성과 LangGraph와의 통합 복잡도로 인해 일부 노드에만 적용
- MultiQuery 생성과 대화 요약에 DSPy `Signature`/`Predict`를 활용하여 프롬프트를 모듈화

### 4. RAGAS 자동 품질 평가
- 응답에 대한 객관적인 지표를 사용자에게 노출하기 위해 사용
- 매 응답마다 Faithfulness, AnswerRelevancy, ContextPrecision 등 지표 계산
- 평가 결과를 State에 저장하여 하이퍼파라미터 최적화의 목적 함수로 활용

### 5. Bayesian Optimization으로 RAG 하이퍼파라미터 튜닝
- RAGAS 지표를 목적 함수로 삼아 Optuna로 `retriever_k`, `reranking_top_k`, `retriever_w`(하이브리드 가중치)를 자동 최적화
- Total Trials: 50회 / Test Dataset: 15개의 복합 질문 시나리오 (구어체 포함)

    |지표|Before|After|개선폭|
    |----|------|-----|-----|
    |충실도|0.69|0.92|**+33.3%**|
    |답변 관련성|0.33|0.38|**+15.1%**|
    |문맥 활용도|0.81|0.81|유지|


## 기술적 판단 및 개선 가능성

### DSPy 적용 범위 (부분 적용)
DSPy로 파이프라인 내 모든 프롬프트를 자동 최적화하는 것이 초기 목표였습니다.  
그러나 자유형 질답 중심의 프로젝트 특성상 Signature 정의가 어렵고, LangGraph State와의 통합 복잡도도 높아 Multi-Query 생성과 대화 요약에만 적용했습니다.

### Multi-Agent (미채택)
LangGraph 기반 Multi-Agent 구현 방법을 검토했으나, 이 프로젝트는 단일 도메인(쿠팡 사고 분석) 질답이 중심이라 역할 분리가 명확히 필요한 상황이 아닙니다.  
현재 구조처럼 단일 에이전트가 Tool Calling으로 웹 검색까지 처리하는 것이 더 적절하다고 판단하여 채택하지 않았습니다.

### Self-Correction (미채택)
이 프로젝트는 응답 품질을 사용자에게 투명하게 보여주기 위해 매 답변 후 RAGAS 평가 점수를 노출합니다.  
여기에 Self-Correction(답변 생성 → 품질 평가 → 기준 미달 시 재생성)을 추가하면 품질을 높일 수 있지만, RAGAS 자체가 내부적으로 LLM을 여러 번 호출하는 구조라 응답 시간이 2배 이상 늘어납니다.  
응답 시간이 더 늘어난다면 포트폴리오 시연 환경에서도 허용하기 어려운 수준이라 판단하여 미구현으로 남겼습니다.

라이브 환경이라면 RAGAS 대신 단일 LLM 호출로 품질을 판단하는 **LLM Judge** 방식으로 평가를 경량화하고, 그 위에 Self-Correction 루프를 얹는 구조가 현실적인 개선 방향입니다.

---

## API 엔드포인트

| Method | 경로 | 설명 |
|---|---|---|
| GET | `/ping` | 서버 상태 확인 |
| GET | `/login` | 사용자 로그인 및 환영 메시지 반환 |
| GET | `/logout` | 사용자 로그아웃 및 대화 메모리 삭제 |
| GET | `/userchat` | AI 에이전트 채팅 (스트리밍) |
