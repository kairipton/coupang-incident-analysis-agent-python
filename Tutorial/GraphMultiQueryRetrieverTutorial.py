"""GraphMultiQueryRetrieverTutorial.py

LangGraph 버전의 "Multi-Query Retrieval" 튜토리얼입니다.

Multi-Query Retrieval이란?
- 사용자의 질문(question)을 그대로 한 번만 검색하면(= 단일 쿼리)
  검색어가 애매하거나 표현이 부족해서 필요한 문서를 놓칠 수 있습니다.
- 그래서 LLM을 이용해 질문을 "검색용 쿼리 여러 개"로 바꿉니다.
- 각 쿼리로 retriever를 여러 번 호출한 뒤 결과를 합쳐(중복 제거) 최종 문서 집합을 만듭니다.

이 튜토리얼의 핵심 로직(3단계)
(A) question -> queries[0..N]   (LLM이 생성)
(B) 각 query로 retriever를 여러 번 실행
(C) 문서 결과를 union(합집합)해서 반환

여기서 LangGraph는 무엇을 해주나?
- 제공 API(MultiQueryRetriever)를 노드 안에서 호출하고,
    LangGraph는 그 노드를 START->END로 배선하는 역할만 합니다.

실행
- Windows PowerShell (프로젝트 루트에서)
  ./.venv/Scripts/python.exe ./Tutorial/GraphMultiQueryRetrieverTutorial.py

주의
- OPENAI_API_KEY가 필요합니다(쿼리 생성 LLM + 벡터 검색 쿼리 임베딩)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

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


def _clean_query_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^\s*\d+\.\s*", "", s)  # "1. foo" -> "foo"
    s = s.strip("\"'")
    return s


def _unique_by_content(docs: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for d in docs:
        key = getattr(d, "page_content", None) or str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _print_docs(docs: list[Any], *, max_docs: int = 3) -> None:
    if not docs:
        print("(0 docs)")
        return
    for i, d in enumerate(docs[:max_docs], start=1):
        text = getattr(d, "page_content", str(d)).replace("\n", " ").strip()
        print(f"- #{i} {text[:160]}")


# ---------------------------------------------------------------------------
# 1) LangGraph State
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    question: str

    # (A) MultiQueryRetriever가 생성한 queries
    queries: list[str]

    # (B) MultiQueryRetriever가 반환한 문서(이미 merge된 결과)
    retrieved_docs_merged: list[Any]


# ---------------------------------------------------------------------------
# 2) 노드: MultiQueryRetriever
# ---------------------------------------------------------------------------


def _normalize_queries(raw: Any) -> list[str]:
    """MultiQueryRetriever 내부 체인의 출력에서 쿼리 문자열 목록을 최대한 안전하게 뽑기."""

    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, dict):
        text = raw.get("text") or raw.get("output") or raw.get("content")
        if text is None:
            return []
        return [x.strip() for x in str(text).splitlines() if x.strip()]
    text = getattr(raw, "content", None)
    if text is None:
        text = str(raw)
    return [x.strip() for x in str(text).splitlines() if x.strip()]


def node_multiquery_retrieve(state: State) -> State:
    """LangChain의 MultiQueryRetriever를 LangGraph 노드 안에서 그대로 사용.

    포인트
    - "멀티쿼리 생성 + 여러 번 검색 + 결과 merge"를 우리가 수동으로 구현하지 않고
      제공되는 컴포넌트(MultiQueryRetriever)에 맡긴다.
    """

    _require_openai_key()

    question = state["question"].strip()
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    retriever = Utils.load_vector_db().as_retriever(k=6)

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "다음 질문을 벡터 검색용으로 4개 쿼리로 바꿔라. 각 쿼리는 한 줄. 쿼리만 출력.\n"
            "질문: {question}"
        ),
    )

    mqr = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=query_prompt,
        include_original=True,
    )

    # 생성된 쿼리 보기(디버그/학습용)
    raw_queries = mqr.llm_chain.invoke({"question": question})
    queries = [_clean_query_line(q) for q in _normalize_queries(raw_queries)]
    queries = [q for q in queries if q]
    # include_original=True라도, 구현/버전에 따라 원 질문이 별도로 보장되지 않을 수 있어
    # 여기서는 확실하게 원 질문을 앞에 넣고 dedup
    queries = [_clean_query_line(question)] + queries

    dedup: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        dedup.append(q)

    # 실제 검색(멀티쿼리 + merge까지 MultiQueryRetriever가 처리)
    docs = list(mqr.invoke(question))
    return {"queries": dedup, "retrieved_docs_merged": docs}


# ---------------------------------------------------------------------------
# 3) 그래프 배선
# ---------------------------------------------------------------------------


builder = StateGraph(State)
builder.add_node("multiquery_retrieve", node_multiquery_retrieve)

builder.add_edge(START, "multiquery_retrieve")
builder.add_edge("multiquery_retrieve", END)

graph = builder.compile()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    print("LangGraph Multi-Query Retriever 튜토리얼")
    print("- 질문을 여러 검색 쿼리로 확장한 뒤, 여러 번 검색해서 결과를 합칩니다.\n")

    while True:
        question = input("질문(Enter로 종료): ").strip()
        if not question:
            break

        state = graph.invoke({"question": question})

        print("\n[생성된 쿼리]")
        for i, q in enumerate(state.get("queries", []), start=1):
            print(f"[{i}] {q}")

        merged = state.get("retrieved_docs_merged", [])
        print(f"\n[최종 검색 결과] (총 {len(merged)}개)")
        _print_docs(merged, max_docs=3)


if __name__ == "__main__":
    main()
