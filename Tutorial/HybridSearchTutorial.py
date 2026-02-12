"""HybridSearchTutorial.py

이 튜토리얼은 LangChain에서 말하는 "Hybrid Search(하이브리드 검색)"를
"조합"으로 구현하는 최소 예제입니다.

Hybrid Search란?
- 키워드 검색(BM25 같은 lexical search) + 벡터 검색(semantic search)을 같이 쓰는 전략
- 둘의 결과를 합치거나(merge/union), 가중치를 줘서 섞거나(rank fusion) 해서
  서로의 약점을 보완합니다.

벡터 검색 vs BM25(키워드) 검색
- 벡터 검색: 질문/문서를 임베딩(숫자 벡터)으로 바꾼 뒤, 벡터 간 "유사도"로 가까운 문서를 찾습니다.
    - 표현이 달라도 의미가 비슷하면 잘 찾는 편(semantic search)
- BM25: 질문의 단어들이 문서에 "얼마나 잘 매칭되는지"를 점수화해 랭킹합니다(lexical search)
    - 자주 등장하는 단어(TF) + 희귀한 단어(IDF) + 문서 길이 보정 등을 이용해 점수를 계산합니다.
    - 에러 코드/설정 키/명령어/정확한 용어처럼 "문자 그대로" 맞춰야 하는 경우에 강합니다.

정리하면
- 벡터 검색은 "의미 유사도"에 강하고,
- BM25는 "키워드(문자열) 매칭"에 강해서,
    둘을 합치면(= 하이브리드) 검색이 더 안정적으로 되는 경우가 많습니다.

이 파일에서 하는 일(핵심 3단계)
1) 벡터 리트리버 준비 (Chroma: Utils.load_vector_db().as_retriever())
2) 키워드 리트리버 준비 (BM25Retriever: tech_manual.md를 chunk로 나눠 인덱싱)
3) 둘을 섞은 Hybrid Retriever 준비 (EnsembleRetriever: Weighted RRF)

그리고 같은 질문을 3가지로 비교합니다.
- Vector only
- BM25 only
- Hybrid(Ensemble)

준비물
- 프로젝트 루트의 .env에 OPENAI_API_KEY (벡터 검색의 쿼리 임베딩에 필요)


BM25 인덱싱(캐시 없음)
- BM25는 임베딩/외부 API 호출이 없어 비용이 들지 않습니다.
- 이 튜토리얼은 가독성을 위해 BM25 캐시(저장/로드)를 사용하지 않고,
  실행할 때마다 (문서 로드 → 청킹 → BM25 인덱싱)을 다시 수행합니다.

실행
- Windows PowerShell:
  ./.venv/Scripts/python.exe ./Tutorial/HybridSearchTutorial.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import openai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 튜토리얼 파일이 ./Tutorial 아래로 이동하면,
    # 기본 import 검색 경로(sys.path)에 프로젝트 루트가 포함되지 않는 경우가 많습니다.
    # 그래서 GameConfig, Utils 같은 "프로젝트 내부 모듈" import가 실패할 수 있어
    # 여기서 루트를 강제로 sys.path에 추가합니다.
    sys.path.insert(0, str(PROJECT_ROOT))

# 문서/벡터DB 경로를 모두 프로젝트 루트 기준으로 상대경로 해석하기 위해
# working directory도 루트로 고정합니다.
os.chdir(PROJECT_ROOT)


from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

import GameConfig as Config
import Utils.Utils as Utils


def _has_openai_key() -> bool:
    """OpenAI API 키 존재 여부만 확인합니다.

    이 튜토리얼에서 Vector/Hybrid 검색은 "쿼리 임베딩"이 필요합니다.
    - Chroma에 저장된 문서를 검색하려면, 사용자의 질문(query)도 임베딩으로 바꿔야 합니다.
    - 그 과정에서 OpenAIEmbeddings가 OpenAI API를 호출합니다.

    그래서 OPENAI_API_KEY가 없으면:
    - BM25 only(키워드 검색)만 수행하고
    - Vector/Hybrid 파트는 안내 메시지를 출력하고 종료합니다.

    Returns:
        bool: OPENAI_API_KEY가 설정되어 있으면 True
    """

    # .env 위치를 "프로젝트 루트"로 고정해두면,
    # 어디서 실행하든(루트/서브폴더/IDE) 같은 키를 읽습니다.
    load_dotenv(PROJECT_ROOT / ".env")
    return bool(os.getenv("OPENAI_API_KEY"))


def _print_docs(title: str, docs: list[Any], *, max_docs: int = 5) -> None:
    """Document 리스트를 사람이 보기 좋게 축약 출력합니다.

    Retriever 결과는 보통 `Document` 객체 리스트이며, 원문 전체를 그대로 출력하면
    터미널이 너무 길어져 학습이 어려워집니다.

    Args:
        title: 섹션 제목(예: "BM25 only")
        docs: Document 리스트
        max_docs: 최대 출력 개수
    """

    # 튜토리얼 출력 가독성용: Document를 전부 출력하면 너무 길어져서
    # 앞부분만 잘라서 보여줍니다.
    print(f"\n[{title}] (총 {len(docs)}개)")
    for i, d in enumerate(docs[:max_docs], start=1):
        text = getattr(d, "page_content", str(d)).replace("\n", " ").strip()
        meta = getattr(d, "metadata", {})
        src = meta.get("source") if isinstance(meta, dict) else None
        head = f"[{i}]" + (f" (source={src})" if src else "")
        print(head, text[:220] + ("..." if len(text) > 220 else ""))


def _build_vector_retriever(k: int = 6):
    """Chroma(Vector DB) 기반의 벡터 retriever를 생성합니다.

    이 프로젝트의 `Utils.load_vector_db()`는 내부적으로:
    - VectorDB가 없으면: 문서를 로드/청킹/임베딩하여 Chroma를 만들고 디스크에 저장
    - VectorDB가 있으면: 디스크의 Chroma를 로드

    즉, 벡터DB 쪽은 이미 "persist -> load" 흐름이 구현되어 있어
    매 실행마다 전체 임베딩을 다시 하지 않습니다.

    Args:
        k: similarity_search에서 상위 몇 개 문서를 가져올지

    Returns:
        VectorStoreRetriever: `vector_db.as_retriever(k=k)`
    """

    # 벡터 리트리버는 이미 Utils.load_vector_db()가 "디스크에 저장된 Chroma"를
    # 로드할 수 있도록 구현되어 있습니다.
    # (없으면 생성, 있으면 로드)
    vector_db = Utils.load_vector_db()
    return vector_db.as_retriever(k=k)


def _build_bm25_retriever(k: int = 6) -> BM25Retriever:
    """튜토리얼용 BM25Retriever를 매번 새로 생성합니다(캐시 없음).

    처리 흐름:
    1) tech_manual.md를 로드
    2) 청킹
    3) 청킹 결과로 BM25 인덱스 생성(BM25Retriever.from_documents)

    Args:
        k: 검색 시 상위 문서 개수

    Returns:
        BM25Retriever: 인메모리 BM25 인덱스를 가진 retriever
    """

    # BM25는 "벡터DB"처럼 별도 서버/스토어에 저장되는 게 아니라,
    # 문서 리스트를 받아서 인메모리 인덱스를 만드는 방식이 일반적입니다.
    # 그래서 이 함수는 (문서 로드 -> 청킹 -> BM25 인덱스 생성)을 한 번에 수행합니다.

    manual_path = PROJECT_ROOT / "Knowledge Base" / Config.manual_md_name

    # chunk 파라미터는 벡터DB 쪽(Utils.load_vector_db)와 맞춰둔 값입니다.
    # (같은 chunk 전략을 쓰면 hybrid 결과 비교가 더 직관적입니다)
    chunk_size = 1000
    chunk_overlap = 200

    loader = TextLoader(str(manual_path), encoding="utf8")
    manual_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(manual_docs)
    for d in chunks:
        d.metadata["category"] = "manual"

    return BM25Retriever.from_documents(chunks, k=k)


def main() -> None:
    """하이브리드 검색을 3가지 방식으로 비교 실행합니다.

    실행 흐름:
    1) BM25 retriever 준비(캐시 로드/생성)
    2) OpenAI 키가 있으면 Vector retriever(Chroma) 준비
    3) Hybrid retriever(EnsembleRetriever)로 결과 융합
    4) 같은 질문을 BM25 / Vector / Hybrid로 각각 검색하여 출력

    참고:
    - EnsembleRetriever는 여러 retriever 결과를 Weighted RRF 방식으로 합칩니다.
        weights=[0.5, 0.5]는 “키워드 vs 벡터” 비중을 동일하게 둔 예시입니다.
    """

    # 한 번만 로드하면 이후 langchain_openai/openai SDK가 동일한 env를 사용합니다.
    # (툴/리트리버 내부에서 추가로 load_dotenv를 호출하지 않아도 됨)
    load_dotenv(PROJECT_ROOT / ".env")
    k = 6

    # Vector/Hybrid 실행 여부를 코드에서 직접 제어합니다.
    # - 튜토리얼에서는 환경변수 플래그보다 "코드 한 줄"이 이해하기 쉽습니다.
    # - Vector/Hybrid를 끄고 BM25만 보려면 False로 바꾸세요.
    enable_vector_and_hybrid = True

    bm25_retriever = _build_bm25_retriever(k=k)

    vector_retriever = None
    hybrid_retriever = None
    if enable_vector_and_hybrid and _has_openai_key():
        # Vector / Hybrid는 "쿼리 임베딩"을 위해 OpenAI API가 필요합니다.
        vector_retriever = _build_vector_retriever(k=k)

        # EnsembleRetriever(앙상블 리트리버)
        # - 여러 retriever의 결과를 "랭크 퓨전(rank fusion)"으로 합칩니다.
        # - 여기서는 Weighted RRF(Reciprocal Rank Fusion) 방식을 사용합니다.
        #     - 각 retriever가 반환한 문서의 순위(rank)에 기반해 점수를 주고,
        #       여러 retriever의 점수를 합산하여 최종 순위를 만듭니다.
        #     - 중요한 포인트: 두 검색기가 점수 체계가 달라도(예: BM25 점수 vs 벡터 유사도)
        #       "순위"만 이용하므로 비교적 안정적으로 합칠 수 있습니다.
        # - weights는 retriever별 기여도를 조절합니다.
        #     - weights=[0.5, 0.5]  => BM25와 Vector를 동일 비중으로 섞음
        #     - 예: weights=[0.7, 0.3] => 키워드(BM25)를 더 신뢰
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )

    try:
        question = input("질문: ").strip()
    except EOFError:
        question = "CPU 과부하"

    if not question:
        return

    # BM25 리트리버로 검색한 결과 6개를 리스트로.
    bm25_docs = list(bm25_retriever.invoke(question))
    _print_docs("BM25 only (키워드)", bm25_docs)

    if not vector_retriever or not hybrid_retriever:
        if not enable_vector_and_hybrid:
            print("\n[안내] 현재 코드 설정(enable_vector_and_hybrid=False)으로 Vector/Hybrid를 비활성화했습니다.")
        else:
            print("\n[안내] Vector/Hybrid는 OPENAI_API_KEY가 필요합니다.")
        return

    try:
        vector_docs = list(vector_retriever.invoke(question))
        hybrid_docs = list(hybrid_retriever.invoke(question))
    except openai.AuthenticationError as e:
        # 키가 없을 때는 위에서 스킵되지만,
        # 키가 "틀렸거나 만료"된 상태면 여기서 인증 오류가 납니다.
        print("\n[안내] OPENAI_API_KEY 인증 실패로 Vector/Hybrid를 건너뜁니다.")
        print(f"에러: {e}")
        return

    _print_docs("Vector only (의미/벡터)", vector_docs)
    _print_docs("Hybrid (BM25 + Vector, Ensemble)", hybrid_docs)


if __name__ == "__main__":
    main()
