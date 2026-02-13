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
- 위 3단계를 "노드"로 쪼개서, 흐름을 눈에 보이게 배선합니다.
  plan_queries -> retrieve_many -> merge

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

    # (A) plan_queries 결과
    queries: list[str]

    # (B) retrieve_many 결과
    per_query_counts: list[int]
    retrieved_docs_raw: list[Any]

    # (C) merge 결과
    retrieved_docs_merged: list[Any]


# ---------------------------------------------------------------------------
# 2) 노드: plan_queries / retrieve_many / merge
# ---------------------------------------------------------------------------


def node_plan_queries(state: State) -> State:
    """LLM으로 질문을 여러 검색 쿼리로 확장.

    초보자 포인트
    - 여기서 LLM은 "답변"을 만드는 게 아니라, "검색어"를 만드는 역할입니다.
    - 좋은 쿼리를 만들수록, retriever가 더 많은 관련 문서를 찾아올 수 있습니다.
    """

    _require_openai_key()

    question = state["question"].strip()

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "다음 질문을 벡터 검색용으로 4개 쿼리로 바꿔라. 각 쿼리는 한 줄. 쿼리만 출력.\n"
            "질문: {question}"
        ),
    )

    # PromptTemplate은 문자열을 만들고, llm.invoke로 호출합니다.
    # 결과 content에는 보통 여러 줄이 들어옵니다.
    msg = llm.invoke(prompt.format(question=question))
    text = getattr(msg, "content", str(msg))

    raw_lines = [x for x in text.splitlines() if x.strip()]

    # include_original=True와 같은 효과: 원 질문도 쿼리 목록에 포함
    queries = [_clean_query_line(question)]
    for line in raw_lines:
        q = _clean_query_line(line)
        if q:
            queries.append(q)

    # 중복 제거(순서 유지)
    dedup: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        dedup.append(q)

    return {"queries": dedup}


def node_retrieve_many(state: State) -> State:
    """생성된 쿼리들을 가지고 retriever를 여러 번 호출.

    초보자 포인트
    - Multi-Query의 핵심은 여기입니다.
    - retriever.invoke(query)를 N번 호출해서 문서들을 넓게 모읍니다(recall 증가).
    """

    _require_openai_key()

    retriever = Utils.load_vector_db().as_retriever(k=6)

    queries = state.get("queries", [])
    counts: list[int] = []
    docs_raw: list[Any] = []

    for q in queries:
        docs_q = list(retriever.invoke(q))
        counts.append(len(docs_q))
        docs_raw.extend(docs_q)

    return {
        "per_query_counts": counts,
        "retrieved_docs_raw": docs_raw,
    }


def node_merge(state: State) -> State:
    """여러 쿼리의 결과를 합치고(Union), 중복을 제거."""

    docs_raw = state.get("retrieved_docs_raw", [])
    merged = _unique_by_content(docs_raw)
    return {"retrieved_docs_merged": merged}


# ---------------------------------------------------------------------------
# 3) 그래프 배선
# ---------------------------------------------------------------------------


builder = StateGraph(State)
builder.add_node("plan_queries", node_plan_queries)
builder.add_node("retrieve_many", node_retrieve_many)
builder.add_node("merge", node_merge)

builder.add_edge(START, "plan_queries")
builder.add_edge("plan_queries", "retrieve_many")
builder.add_edge("retrieve_many", "merge")
builder.add_edge("merge", END)

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

        print("\n[각 쿼리별 retriever 호출 결과(= 여러 번 검색됨)]")
        counts = state.get("per_query_counts", [])
        for i, (q, c) in enumerate(zip(state.get("queries", []), counts), start=1):
            print(f"- #{i} docs={c}  query={q}")

        raw = state.get("retrieved_docs_raw", [])
        merged = state.get("retrieved_docs_merged", [])
        print(f"\n(합치기 전 raw={len(raw)}개, 중복 제거 후 union={len(merged)}개)")

        print("\n[최종 검색 결과(중복 제거 후)]")
        _print_docs(merged, max_docs=3)


if __name__ == "__main__":
    main()
