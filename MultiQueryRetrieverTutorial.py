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
    ./.venv/Scripts/python.exe ./MultiQueryRetrieverTutorial.py
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import Utils.Utils as Utils


# ---------------------------------------------------------------------------
# 0) 공통: LLM / Vector DB 준비
# ---------------------------------------------------------------------------


def build_llm() -> ChatOpenAI:
    """LLM(여기서는 OpenAI Chat 모델)을 준비합니다.

    MultiQueryRetriever는 "질문 -> 여러 검색 쿼리"를 만들어내기 위해 LLM이 필요합니다.
    실제 문서 검색은 retriever가 수행하지만, "어떤 쿼리로 검색할지"는 LLM이 결정합니다.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY가 필요합니다.")

    # 모델 이름은 .env의 OPENAI_MODEL로 바꿀 수 있습니다.
    # 예: OPENAI_MODEL=gpt-4o-mini
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def print_docs(docs: list[Any], max_docs: int = 5) -> None:
    """검색 결과 문서를 보기 좋게 출력하는 헬퍼입니다.

    중요: MultiQueryRetriever가 동작하는 데 필수는 아닙니다.
    - 없어도 docs = mqr.invoke(question) 까지는 동일하게 동작합니다.
    - 다만 Document 객체를 그대로 print하면 너무 길고 지저분해져서, 튜토리얼 가독성을 위해 둔 함수입니다.
    """

    if not docs:
        print("(0 docs)")
        return
    for i, d in enumerate(docs[:max_docs], start=1):
        text = getattr(d, "page_content", str(d)).replace("\n", " ")
        meta = getattr(d, "metadata", {})
        src = meta.get("source") if isinstance(meta, dict) else None
        head = f"[{i}]" + (f" (source={src})" if src else "")
        print(head, text[:220] + ("..." if len(text) > 220 else ""))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    # 1) LLM 준비
    llm = build_llm()

    # 2) "기본 retriever" 준비
    # - 이 프로젝트는 Chroma(Vector DB)를 Utils.load_vector_db()로 로드합니다.
    # - as_retriever(k=6): 각 검색에서 상위 6개 문서 조각을 가져오게 합니다.
    retriever = Utils.load_vector_db().as_retriever(k=6)

    # 3) "질문을 여러 검색 쿼리로 확장"하는 프롬프트
    # - MultiQueryRetriever 내부에서는 대략 다음 파이프라인을 만듭니다:
    #     llm_chain = prompt | llm | output_parser
    # - langchain-classic의 MultiQueryRetriever는 기본적으로 "한 줄에 하나의 쿼리"를 기대합니다.
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "다음 질문을 벡터 검색용으로 4개 쿼리로 바꿔라. 각 쿼리는 한 줄. 쿼리만 출력.\n"
            "질문: {question}"
        ),
    )

    # 4) MultiQueryRetriever 생성
    # - retriever: 실제 벡터 검색을 수행하는 객체
    # - llm: 질문을 여러 쿼리로 "생성"하는 모델
    # - prompt: 쿼리 생성 규칙
    # - include_original=True:
    #     생성된 쿼리들 + 원래 질문(question)도 함께 검색에 포함합니다.
    #     (LLM이 질문을 엉뚱하게 바꿔버리는 경우를 완화하는 안전장치 역할도 합니다.)
    mqr = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=query_prompt,
        include_original=True,
    )

    # 5) 질문 입력 (한 번만 실행)
    # - 튜토리얼은 "한 질문에 대해 MQR이 어떤 쿼리를 만들고, 어떤 결과를 내는지"를 보는 것이 목적이라
    #   반복 입력 루프를 두지 않고 1회 실행 후 종료합니다.
    # - stdin이 닫혀 있는 환경(파이프/리다이렉션/자동 실행)에서도 죽지 않도록 기본값을 둡니다.
    try:
        question = input("질문: ").strip()
    except EOFError:
        question = "weird"

    if not question:
        return

    print("\n[생성된 쿼리]")

    # mqr.llm_chain은 "질문 -> 쿼리 리스트"를 만드는 Runnable 입니다.
    # 아래 출력은 "MultiQueryRetriever가 검색에 사용할 쿼리"를 눈으로 확인하려는 목적입니다.
    # (이 부분 역시 학습을 위한 출력이며, 없어도 검색 자체는 됩니다.)
    q = mqr.llm_chain.invoke({"question": question})
    print(q)

    print("\n[검색 결과]")

    # mqr.invoke(question)이 실제 핵심 호출입니다.
    # 내부적으로는 대략 다음을 수행합니다:
    #  - LLM으로 여러 쿼리 생성
    #  - 각 쿼리로 retriever.invoke(query) 여러 번 호출
    #  - 문서 리스트를 합치고(중복 제거) 반환
    docs = mqr.invoke(question)
    print(f"(총 {len(docs)}개)")
    print_docs(docs, max_docs=6)


if __name__ == "__main__":
    main()
