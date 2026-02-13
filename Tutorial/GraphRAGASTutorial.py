"""GraphRAGASTutorial.py

LangGraph 버전의 RAGAS 평가 튜토리얼입니다.

RAGAS가 필요한 이유(요약)
- RAG는 "검색(retrieve)" + "생성(generate)"이 붙어있는 구조라서
  답변이 그럴듯해 보여도 실제로는:
  - 검색이 엉뚱했거나
  - 컨텍스트에 없는 내용을 지어냈거나(환각)
  - 질문과 무관한 답을 했을 수 있습니다.

그래서 (question, answer, contexts) 샘플을 모아
- faithfulness: 컨텍스트에 근거했는가?
- answer_relevancy: 질문에 답했는가?
같은 지표로 "진단"합니다.

이 파일에서 LangGraph를 쓰는 이유
- RAG 파이프라인을 "노드"로 나눠서 흐름을 명확히 하기 위해서입니다.
  build_samples -> (optional) evaluate

주의(비용)
- RAG 답변 생성에 LLM 호출이 들어갑니다.
- RAGAS 평가 자체도 LLM/Embeddings 호출이 들어갈 수 있어 비용이 추가됩니다.

실행
- Windows PowerShell (프로젝트 루트에서)
  ./.venv/Scripts/python.exe ./Tutorial/GraphRAGASTutorial.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 0) 프로젝트 루트 고정
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

import Utils.Utils as Utils


def _require_openai_key() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 필요합니다. (./.env에 설정)")


def build_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


def rag_answer(question: str, *, retriever, llm: ChatOpenAI) -> dict[str, Any]:
    """retrieve -> prompt -> answer 형태의 아주 단순한 RAG.

    초보자 포인트
    - RAGAS 평가를 하려면 최소한 아래 3가지가 필요합니다.
      1) question: 질문
      2) contexts: 검색으로 가져온 근거 텍스트들
      3) answer: 그 근거를 바탕으로 생성한 답변

    - 여기서는 '좋은 답변'을 만들기보다,
      평가 입력 형식을 만드는 게 목적이라 구조를 단순화합니다.
    """

    docs = list(retriever.invoke(question))
    contexts = [d.page_content for d in docs]

    context_text = "\n\n".join(contexts)[:8000]

    prompt = (
        "너는 시스템 관리자 매뉴얼을 바탕으로 질문에 답하는 도우미야.\n"
        "반드시 아래 [컨텍스트]에 근거해서만 답해.\n\n"
        "[컨텍스트]\n"
        f"{context_text}\n\n"
        "[질문]\n"
        f"{question}\n\n"
        "[답변]"
    )

    answer_msg = llm.invoke(prompt)
    answer = getattr(answer_msg, "content", str(answer_msg))

    return {"question": question, "answer": answer, "contexts": contexts}


def evaluate_with_ragas(samples: list[dict[str, Any]]):
    """RAGAS로 samples(question/answer/contexts)를 평가합니다."""

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    dataset = Dataset.from_list(samples)

    llm = build_llm()
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)

    return evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )


# ---------------------------------------------------------------------------
# 1) LangGraph State
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    questions: list[str]
    samples: list[dict[str, Any]]

    run_ragas: bool
    ragas_result: Any


# ---------------------------------------------------------------------------
# 2) 노드: build_samples / evaluate
# ---------------------------------------------------------------------------


def node_build_samples(state: State) -> State:
    """질문 리스트를 받아 RAG 샘플(question/answer/contexts)을 만듭니다."""

    _require_openai_key()

    retriever = Utils.load_vector_db().as_retriever(k=6)
    llm = build_llm()

    samples: list[dict[str, Any]] = []
    for q in state.get("questions", []):
        samples.append(rag_answer(q, retriever=retriever, llm=llm))

    return {"samples": samples}


def node_evaluate(state: State) -> State:
    """RAGAS로 평가합니다. (비용 발생 가능)"""

    samples = state.get("samples", [])
    result = evaluate_with_ragas(samples)
    return {"ragas_result": result}


def should_run_ragas(state: State) -> str:
    """조건부 분기: run_ragas=True면 evaluate로, 아니면 END로."""

    return "evaluate" if state.get("run_ragas", True) else "end"


# ---------------------------------------------------------------------------
# 3) 그래프 배선
# ---------------------------------------------------------------------------


builder = StateGraph(State)
builder.add_node("build_samples", node_build_samples)
builder.add_node("evaluate", node_evaluate)

builder.add_edge(START, "build_samples")

# build_samples 후에 "평가를 할지 말지"를 조건으로 분기합니다.
builder.add_conditional_edges(
    "build_samples",
    should_run_ragas,
    {
        "evaluate": "evaluate",
        "end": END,
    },
)

builder.add_edge("evaluate", END)

graph = builder.compile()


def main() -> None:
    _require_openai_key()

    # 튜토리얼용: 질문 샘플은 2~3개로 시작하는 게 비용/시간 면에서 안전합니다.
    questions = [
        "CPU 과부하 상태에서 어떤 조치를 해야 해?",
        "디스크가 가득 찼을 때(용량 부족) 어떤 절차로 해결해?",
        "메모리 누수처럼 보이는데 실제로는 디스크 문제일 수도 있을까?",
    ]

    # 평가까지 돌릴지 여부(튜토리얼용 스위치)
    run_ragas = True

    state = graph.invoke({"questions": questions, "run_ragas": run_ragas})

    samples = state.get("samples", [])
    print("\n[RAG 답변 샘플]")
    for i, s in enumerate(samples, start=1):
        print(f"\n[{i}] Q: {s['question']}")
        print(f"A: {s['answer'][:400]}" + ("..." if len(s["answer"]) > 400 else ""))
        print(f"contexts: {len(s['contexts'])}개")

    if not run_ragas:
        print("\n[안내] run_ragas=False 이므로 평가를 건너뜁니다.")
        return

    print("\n[RAGAS 평가 실행] (LLM/Embeddings 호출이 추가로 발생할 수 있습니다)")
    result = state.get("ragas_result")

    try:
        print(f"\n[RAGAS 평가 RAW 결과] -> {result}")
        if hasattr(result, "to_pandas"):
            print("\n[평가 결과 테이블]")
            print(result.to_pandas())
    except Exception:
        print("평가 결과 출력 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main()
