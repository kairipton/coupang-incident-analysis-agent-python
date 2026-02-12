"""MultiQueryRetrieverTutorial.py

이 파일은 "MultiQueryRetriever가 실제로 어떻게 동작하는지"를 코드로 확인하기 위한 튜토리얼입니다.

핵심 아이디어
1) 사용자의 질문(question)을 그대로 벡터 검색에 던지면(= 단일 쿼리) 정보가 부족해서 문서가 덜 걸릴 수 있습니다.
2) MultiQueryRetriever는 LLM을 이용해 question을 "검색용 쿼리 여러 개"로 확장합니다.
3) 각 쿼리로 retriever를 여러 번 호출하고, 결과 문서들을 합쳐(중복 제거) 최종 결과를 만듭니다.

즉, 동작 흐름은 아래 3단계입니다.
    (A) question -> [query1, query2, query3, query4]  (LLM이 생성)
    (B) retriever(query_i) 를 여러 번 실행
    (C) 문서 결과를 union(합집합)해서 반환

준비물
- 프로젝트 루트의 .env에 OPENAI_API_KEY
- 이 프로젝트의 벡터DB 로더: Utils.load_vector_db()

실행
- Windows PowerShell:
    ./.venv/Scripts/python.exe ./Tutorial/MultiQueryRetrieverTutorial.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


from dotenv import load_dotenv

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import Utils.Utils as Utils


def build_llm() -> ChatOpenAI:
    """LLM(여기서는 OpenAI Chat 모델)을 준비합니다."""

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY가 필요합니다.")

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def print_docs(docs: list[Any], max_docs: int = 5) -> None:
    """검색 결과 문서를 보기 좋게 출력하는 헬퍼입니다."""

    if not docs:
        print("(0 docs)")
        return
    for i, d in enumerate(docs[:max_docs], start=1):
        text = getattr(d, "page_content", str(d)).replace("\n", " ")
        meta = getattr(d, "metadata", {})
        src = meta.get("source") if isinstance(meta, dict) else None
        head = f"[{i}]" + (f" (source={src})" if src else "")
        print(head, text[:220] + ("..." if len(text) > 220 else ""))


def _clean_query_line(line: str) -> str:
    """LLM이 만든 '검색 쿼리'를 사람이 보기 좋게 정리합니다."""

    s = line.strip()
    s = re.sub(r"^\s*\d+\.\s*", "", s)  # "1. foo" -> "foo"
    s = s.strip("\"'")
    return s


def _unique_by_content(docs: list[Any]) -> list[Any]:
    """문서 리스트에서 중복을 단순 제거합니다(page_content 기준)."""

    out: list[Any] = []
    seen: set[str] = set()
    for d in docs:
        key = getattr(d, "page_content", None) or str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def main() -> None:
    llm = build_llm()

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

    try:
        question = input("질문: ").strip()
    except EOFError:
        question = "weird"

    if not question:
        return

    print("\n[생성된 쿼리]")
    raw_queries = list(mqr.llm_chain.invoke({"question": question}))
    queries = [_clean_query_line(question)] + [
        _clean_query_line(x) for x in raw_queries if _clean_query_line(x)
    ]
    for i, query in enumerate(queries, start=1):
        print(f"[{i}] {query}")

    print("\n[각 쿼리별 retriever 호출 결과(= 실제로 여러 번 검색됨)]")
    per_query_docs: list[Any] = []
    for i, query in enumerate(queries, start=1):
        docs_i = list(retriever.invoke(query))
        per_query_docs.extend(docs_i)
        print(f"- #{i} docs={len(docs_i)}  query={query}")
    manual_union = _unique_by_content(per_query_docs)
    print(f"(직접 합친 결과: raw={len(per_query_docs)}개, unique_union={len(manual_union)}개)")

    print("\n[검색 결과]")
    docs = mqr.invoke(question)
    print(f"(총 {len(docs)}개)")
    print_docs(docs, max_docs=6)


if __name__ == "__main__":
    main()
