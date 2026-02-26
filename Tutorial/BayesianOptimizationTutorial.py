"""
Optuna 베이지안 최적화 튜토리얼
================================
목표: 하이브리드 리트리버의 파라미터(BM25 가중치, k, rerank top_k)를
      RAGAS 점수를 기준으로 자동 탐색하여 최적값을 찾는다.

최적화 대상 파라미터:
    - w_bm25      : BM25 vs Vector 가중치 (0.1 ~ 0.9)
    - k           : 각 리트리버가 가져올 문서 수 (3 ~ 12)
    - top_k_rerank: Cross-Encoder 리랭킹 후 최종 사용할 문서 수 (3 ~ 8)

평가 메트릭 (정답 없이 측정 가능한 것들):
    - faithfulness       : 답변이 검색된 문서에 근거하는가 (가중치 40%)
    - answer_relevancy   : 답변이 질문에 관련 있는가 (가중치 30%)
    - context_utilization: 검색된 문서를 잘 활용했는가 (가중치 30%)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import dotenv
dotenv.load_dotenv()

import optuna
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, AIMessage
from sentence_transformers import CrossEncoder
from ragas import evaluate
from ragas.metrics import faithfulness, ContextUtilization
from ragas.metrics import AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from datasets import Dataset

import Utils.Utils as Utils
import GameConfig as Config


# ----------------------------------------------------------------
# 테스트 질문셋 (정답 없음 - 정답 없이 측정 가능한 메트릭만 사용)
# ----------------------------------------------------------------
TEST_QUESTIONS = [
    "쿠팡에서 유출된 개인정보의 종류는 무엇인가요?",
    "쿠팡 개인정보 유출 사건의 피해 규모는 어느 정도인가요?",
    "쿠팡은 개인정보 유출 사건에 어떻게 대응했나요?",
    "쿠팡 유출 사건에서 해커는 어떤 방식으로 정보를 탈취했나요?",
    "쿠팡 개인정보 유출로 인한 법적 처벌이나 과징금은 얼마인가요?",
]


# ----------------------------------------------------------------
# 전역 리소스 (trial마다 재생성 방지)
# ----------------------------------------------------------------
print("[초기화] 문서 로딩 및 리소스 준비 중...")

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
_all_chunks = _splitter.split_documents(Utils.get_documents())
for c in _all_chunks:
    c.metadata["category"] = "docs"

_vector_db = Utils.load_vector_db()
_cross_encoder = CrossEncoder(Config.cross_encoder_rerank_model_name, device=Config.cross_encoder_device)
_llm = ChatOpenAI(model=Config.llm_model_name)
_eval_llm = ChatOpenAI(model=Config.llm_model_name, temperature=1)
_embeddings = OpenAIEmbeddings(model=Config.embedding_model_name)

print("[초기화] 완료\n")


def _build_retriever(w_bm25: float, k: int) -> EnsembleRetriever:
    """
    파라미터를 받아서 하이브리드 리트리버를 생성.
    trial마다 다른 가중치/k로 리트리버를 새로 만들어서 성능을 비교한다.
    """
    bm25 = BM25Retriever.from_documents(_all_chunks, k=k)
    vector = _vector_db.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[w_bm25, 1.0 - w_bm25],  # BM25 + Vector 가중치 합이 1.0이 되도록
    )


def _search_and_rerank(question: str, retriever: EnsembleRetriever, top_k: int) -> list[str]:
    """
    하이브리드 검색 후 Cross-Encoder로 리랭킹.
    리트리버가 가져온 문서 후보들을 Cross-Encoder로 재점수화해서
    질문과 가장 관련 높은 top_k개만 돌려준다.
    """
    # 1. 하이브리드 검색으로 문서 후보 가져오기
    docs = retriever.invoke(question)
    contents = [d.page_content for d in docs if hasattr(d, "page_content")]

    if not contents:
        return []

    # 2. Cross-Encoder로 (질문, 문서) 쌍을 점수화
    #    - bi-encoder(임베딩)와 달리 질문+문서를 같이 보고 직접 관련도를 계산
    #    - 점수가 높을수록 질문과 관련성이 높음
    pairs = [(question, doc) for doc in contents]
    scores = [float(s) for s in _cross_encoder.predict(pairs)]

    # 3. 점수 기준 내림차순 정렬 후 상위 top_k개만 반환
    ranked = sorted(zip(scores, contents), key=lambda x: x[0], reverse=True)
    return [text for (_, text) in ranked[:top_k]]


def _generate_answer(question: str, contexts: list[str]) -> str:
    """검색 결과를 바탕으로 LLM 답변 생성"""
    prompt = f"""
        당신은 쿠팡 사태의 모든것을 알고 있는 전문가입니다.
        주어진 질문과 정보를 바탕으로 답변하세요.

        [질문]
        {question}

        [정보]
        {chr(10).join(contexts)}
    """
    response = _llm.invoke([SystemMessage(content=prompt)])
    return getattr(response, "content", str(response))


def _run_ragas(question: str, answer: str, contexts: list[str]) -> float:
    """
    RAGAS 평가 실행 후 종합 점수(0.0 ~ 1.0) 반환.
    세 가지 메트릭을 가중 평균해서 하나의 점수로 합산한다.
    """
    if not contexts or not answer:
        return 0.0

    # RAGAS가 요구하는 입력 형식으로 변환
    data = {
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)

    metrics = [
        faithfulness,               # 답변이 검색 문서에 근거하는가 (환각 탐지)
        AnswerRelevancy(strictness=1),  # 답변이 질문에 관련 있는가
        ContextUtilization(),       # 검색된 문서를 얼마나 잘 활용했는가
    ]

    try:
        # return_executor=True: 결과 객체 대신 Executor를 받아서
        # ragas 내부 버그(IndexError)를 우회함
        executor = evaluate(
            dataset,
            metrics=metrics,
            llm=LangchainLLMWrapper(_eval_llm, bypass_temperature=True),
            embeddings=LangchainEmbeddingsWrapper(_embeddings),
            raise_exceptions=False,
            return_executor=True,
        )
        raw = executor.results()  # [faithfulness값, relevancy값, utilization값]
    except Exception as e:
        print(f"  [RAGAS 오류] {e}")
        return 0.0

    if isinstance(raw, list) and len(raw) == len(metrics):
        # 세 메트릭을 가중 평균해서 하나의 점수로 합산
        # faithfulness 40% + answer_relevancy 30% + context_utilization 30%
        weights = [0.4, 0.3, 0.3]
        score = sum(float(v) * w for v, w in zip(raw, weights) if v is not None)
        return score

    return 0.0


# ----------------------------------------------------------------
# Optuna 목적 함수
# ----------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """
    Optuna가 반복 호출하는 목적 함수.
    한 번 호출 = 한 번의 trial (파라미터 조합 하나를 시도).

    베이지안 최적화 흐름:
        1. 처음 몇 번은 랜덤하게 파라미터를 고름 (탐색)
        2. 이후에는 지금까지 높은 점수를 냈던 파라미터 근처를 집중 탐색 (활용)
        3. 이를 반복해서 점점 좋은 파라미터로 수렴
    """
    # trial.suggest_*: Optuna가 지정한 범위 안에서 파라미터 값을 제안
    # - 처음에는 랜덤, 이후엔 이전 결과를 참고해서 좋아 보이는 값을 제안
    w_bm25   = trial.suggest_float("w_bm25", 0.1, 0.9)     # BM25 가중치
    k        = trial.suggest_int("k", 3, 12)                # 검색 문서 수
    top_k    = trial.suggest_int("top_k_rerank", 3, 8)      # 리랭킹 후 최종 사용 수

    print(f"\n[Trial {trial.number}] w_bm25={w_bm25:.2f}, k={k}, top_k_rerank={top_k}")

    # 이번 trial의 파라미터로 리트리버 생성
    retriever = _build_retriever(w_bm25, k)

    # 모든 테스트 질문에 대해 검색 → 답변 → 평가를 수행하고 평균 점수를 반환
    scores = []
    for q in TEST_QUESTIONS:
        contexts = _search_and_rerank(q, retriever, top_k)  # 검색 + 리랭킹
        answer   = _generate_answer(q, contexts)             # LLM 답변 생성
        score    = _run_ragas(q, answer, contexts)           # RAGAS 평가
        print(f"  질문: {q[:30]}...  점수: {score:.4f}")
        scores.append(score)

    # 평균 점수를 반환 → Optuna는 이 값을 최대화하는 파라미터를 찾음
    avg = sum(scores) / len(scores)
    print(f"  → Trial {trial.number} 평균 점수: {avg:.4f}")
    return avg


# ----------------------------------------------------------------
# 최적화 실행
# ----------------------------------------------------------------
if __name__ == "__main__":

    # optuna 로그 레벨 조정 (WARNING 이상만 출력)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",           # 점수를 최대화하는 방향으로 탐색
        sampler=optuna.samplers.TPESampler(seed=42),  # TPE = Tree-structured Parzen Estimator
                                                      # 베이지안 최적화의 일종으로,
                                                      # 좋은 결과를 낸 파라미터 분포를 모델링해서
                                                      # 다음 trial을 제안하는 알고리즘
        study_name="hybrid_retriever_optimization",
        storage="sqlite:///optuna_results.db",  # 결과를 SQLite에 저장
        load_if_exists=True,                    # 이미 있으면 이어서 실행
    )

    print("=" * 60)
    print("베이지안 최적화 시작")
    print(f"테스트 질문 수: {len(TEST_QUESTIONS)}")
    print("=" * 60)

    study.optimize(
        objective,
        n_trials=10,        # trial 수 (비용과 트레이드오프)
        show_progress_bar=False,
    )

    # ----------------------------------------------------------------
    # 결과 출력
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("최적화 완료")
    print("=" * 60)
    print(f"최고 점수  : {study.best_value:.4f}")
    print(f"최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"  {k:20s} = {v}")

    print("\n[전체 Trial 결과 요약]")
    df = study.trials_dataframe()
    print(df[["number", "value", "params_w_bm25", "params_k", "params_top_k_rerank"]].to_string(index=False))
