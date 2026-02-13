"""GraphHybridSearchTutorial.py

LangGraph 버전의 "Hybrid Search(하이브리드 검색)" 튜토리얼입니다.

이 튜토리얼의 목적
- BM25(키워드 검색)와 Vector Search(의미 기반 검색)를 "둘 다" 실행해 보고
- 결과를 단순 비교하거나, 두 결과를 섞어(Ensemble) 보는 흐름을 이해합니다.
- 그리고 그 흐름을 LangGraph(StateGraph)로 "노드 단위"로 나눠서 배선합니다.

중요 포인트 (핵심 로직)
1) BM25는 로컬 인덱스(비용 0)라서 키가 없어도 동작
2) Vector Search는 쿼리 임베딩에 OpenAI API가 필요(OPENAI_API_KEY)
3) Hybrid는 BM25 + Vector 결과를 rank fusion(여기서는 EnsembleRetriever/Weighted RRF)으로 결합

실행
- Windows PowerShell (프로젝트 루트에서)
  ./.venv/Scripts/python.exe ./Tutorial/GraphHybridSearchTutorial.py

그래프 구조
+-----------+  
| __start__ |
+-----------+
       *
       *
       *
+------------+
| detect_key |
+------------+
       *
       *
       *
   +------+
   | bm25 |
   +------+
       *
       *
       *
  +--------+
  | vector |
  +--------+
       *
       *
       *
  +--------+
  | hybrid |
  +--------+
       *
       *
       *
  +-------+
  | print |
  +-------+
       *
       *
       *
  +---------+
  | __end__ |
  +---------+

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, START, END

# LangChain retrievers
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# 0) 프로젝트 루트 고정: import/.env/상대경로가 흔들리지 않게
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

import GameConfig as Config
import Utils.Utils as Utils


def _has_openai_key() -> bool:
    load_dotenv(PROJECT_ROOT / ".env")
    return bool(os.getenv("OPENAI_API_KEY"))


def _print_docs(title: str, docs: list[Any], *, max_docs: int = 3) -> None:
    """튜토리얼용 최소 출력.

    - '예쁘게' 출력하기보다, 어떤 검색기가 어떤 내용의 문서를 몇 개 찾았는지만 빠르게 확인합니다.
    """

    print(f"\n[{title}] docs={len(docs)}")
    for i, d in enumerate(docs[:max_docs], start=1):
        text = getattr(d, "page_content", str(d)).replace("\n", " ").strip()
        print(f"- #{i} {text[:160]}")


# ---------------------------------------------------------------------------
# 1) BM25 / Vector / Hybrid 리트리버 준비 (그래프 밖에서 준비해도 OK)
#    - 그래프는 "흐름"을 보여주는 것이 목적이므로, 준비 코드는 분리해둡니다.
# ---------------------------------------------------------------------------


def build_bm25_retriever(*, k: int = 6) -> BM25Retriever:
    """tech_manual.md를 청킹하고, 그 청킹 결과로 BM25 인덱스를 만들어 retriever를 반환."""

    manual_path = PROJECT_ROOT / "Knowledge Base" / Config.manual_md_name
    loader = TextLoader(str(manual_path), encoding="utf8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for d in chunks:
        d.metadata["category"] = "manual"

    return BM25Retriever.from_documents(chunks, k=k)


def build_vector_retriever(*, k: int = 6):
    """Chroma(VectorDB) 기반 vector retriever. 쿼리 임베딩 때문에 OPENAI_API_KEY가 필요."""

    vector_db = Utils.load_vector_db()
    return vector_db.as_retriever(k=k)


def build_hybrid_retriever(*, bm25_retriever, vector_retriever):
    """BM25 + Vector 결과를 rank fusion(Weighted RRF)으로 결합."""

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


# ---------------------------------------------------------------------------
# 2) LangGraph State: 질문 + 각 검색기의 결과를 담아 전달
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    question: str

    # 실행 제어
    enable_vector_and_hybrid: bool
    has_openai_key: bool

    # 결과
    bm25_docs: list[Any]
    vector_docs: list[Any]
    hybrid_docs: list[Any]


# ---------------------------------------------------------------------------
# 3) LangGraph 노드들
#    - 노드는 "입력 state → 출력 업데이트"만 담당합니다.
# ---------------------------------------------------------------------------


def node_detect_key(state: State) -> State:
    """OPENAI_API_KEY 존재 여부를 state에 기록."""

    return {"has_openai_key": _has_openai_key()}


def node_bm25_search(state: State) -> State:
    """BM25 검색은 키가 없어도 항상 가능합니다."""

    question = state["question"]
    bm25 = build_bm25_retriever(k=6)
    docs = list(bm25.invoke(question))
    return {"bm25_docs": docs}


def node_vector_search(state: State) -> State:
    """Vector 검색: 키가 없으면 실행할 수 없으니 빈 결과로 둡니다."""

    if not state.get("enable_vector_and_hybrid", True):
        return {"vector_docs": []}
    if not state.get("has_openai_key", False):
        return {"vector_docs": []}

    question = state["question"]
    vector = build_vector_retriever(k=6)
    docs = list(vector.invoke(question))
    return {"vector_docs": docs}


def node_hybrid_search(state: State) -> State:
    """Hybrid 검색: BM25 + Vector를 결합. 키 없으면 스킵."""

    if not state.get("enable_vector_and_hybrid", True):
        return {"hybrid_docs": []}
    if not state.get("has_openai_key", False):
        return {"hybrid_docs": []}

    question = state["question"]

    bm25 = build_bm25_retriever(k=6)
    vector = build_vector_retriever(k=6)
    hybrid = build_hybrid_retriever(bm25_retriever=bm25, vector_retriever=vector)

    docs = list(hybrid.invoke(question))
    return {"hybrid_docs": docs}


def node_print(state: State) -> State:
    """튜토리얼은 '눈으로 확인'이 중요하니 결과를 출력."""

    _print_docs("BM25 only (키워드)", state.get("bm25_docs", []))

    if not state.get("enable_vector_and_hybrid", True):
        print("\n[안내] enable_vector_and_hybrid=False 이므로 Vector/Hybrid를 건너뜁니다.")
        return {}

    if not state.get("has_openai_key", False):
        print("\n[안내] Vector/Hybrid는 OPENAI_API_KEY가 필요합니다. (.env 설정)")
        return {}

    _print_docs("Vector only (의미/벡터)", state.get("vector_docs", []))
    _print_docs("Hybrid (BM25 + Vector, Ensemble)", state.get("hybrid_docs", []))
    return {}


# ---------------------------------------------------------------------------
# 4) 그래프 배선 (START -> keycheck -> bm25 -> vector -> hybrid -> print -> END)
# ---------------------------------------------------------------------------


builder = StateGraph(State)
builder.add_node("detect_key", node_detect_key)
builder.add_node("bm25", node_bm25_search)
builder.add_node("vector", node_vector_search)
builder.add_node("hybrid", node_hybrid_search)
builder.add_node("print", node_print)

builder.add_edge(START, "detect_key")
builder.add_edge("detect_key", "bm25")
builder.add_edge("bm25", "vector")
builder.add_edge("vector", "hybrid")
builder.add_edge("hybrid", "print")
builder.add_edge("print", END)

graph = builder.compile()

print( graph.get_graph().draw_ascii() )

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    print("LangGraph Hybrid Search 튜토리얼")
    print("- BM25 / Vector / Hybrid 결과를 비교합니다.")
    print("- 예: 'CPU 과부하' / '디스크 용량 부족'\n")

    enable_vector_and_hybrid = True

    while True:
        question = input("질문(Enter로 종료): ").strip()
        if not question:
            break

        graph.invoke(
            {
                "question": question,
                "enable_vector_and_hybrid": enable_vector_and_hybrid,
            }
        )


if __name__ == "__main__":
    pass
    main()
