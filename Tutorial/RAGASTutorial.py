"""RAGASTutorial.py

이 튜토리얼은 "내 RAG가 괜찮은가?"를 정량적으로 체크하는
RAGAS(RAG Assessment) 최소 예제입니다.

RAGAS를 왜 쓰나?
- RAG는 "검색(retrieve)"과 "생성(generate)"이 합쳐진 구조라서,
    단순히 답변이 그럴듯해 보이는지만으로 품질을 판단하기 어렵습니다.
- 그래서 (질문 → 검색된 컨텍스트 → 생성된 답변) 흐름을 샘플로 모은 뒤,
    특정 관점의 지표(metrics)로 점수화해서 개선 방향을 잡습니다.

이 튜토리얼에서 보는 대표 지표(최소 2개)
- faithfulness(충실성): 답변이 컨텍스트에 근거했는가? (환각/지어냄 감소)
- answer_relevancy(답변 관련성): 질문에 "제대로" 답했는가?

중요: RAGAS는 "정답"이 아니라 "진단"에 가깝습니다.
- 점수가 낮으면: (검색이 틀림 / 컨텍스트가 부족 / 프롬프트가 애매 / 모델이 과도하게 추론)
    같은 원인을 의심하고 실험을 설계하는 데 도움이 됩니다.

핵심 아이디어
- RAG 파이프라인(= Retriever로 컨텍스트 뽑고 LLM으로 답변 생성)을 한 번 만든 뒤
- (질문, 답변, 컨텍스트) 형태의 샘플들을 모아서
- RAGAS metrics로 평가합니다.

주의(비용)
- BM25는 로컬 연산이라 API 비용이 없습니다.
- 하지만 이 튜토리얼은 "Vector RAG"를 사용하고, 또한 RAGAS 평가 자체도 LLM 호출이 들어갑니다.
    즉, OPENAI_API_KEY가 있어야 하고 API 비용이 발생할 수 있습니다.

이 파일에서 하는 일(큰 흐름)
1) Vector retriever 준비 
2) (질문 → 컨텍스트 → 답변) 샘플 생성
3) RAGAS로 샘플 평가 후 점수 출력

실행
- Windows PowerShell:
  ./.venv/Scripts/python.exe ./Tutorial/RAGASTutorial.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 프로젝트 내부 모듈 경로/상대경로를 안정적으로 쓰기 위해 cwd를 루트로 고정
os.chdir(PROJECT_ROOT)

from langchain_openai import ChatOpenAI

import Utils.Utils as Utils


def _require_openai_key() -> None:
    """이 튜토리얼은 Vector RAG + RAGAS 평가로 OpenAI 호출이 필요합니다."""

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 필요합니다. (./.env에 설정)")


def build_llm() -> ChatOpenAI:
    """RAG 답변 생성 및 RAGAS 평가에 사용할 LLM."""

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


def rag_answer(question: str, *, retriever, llm: ChatOpenAI, k: int = 6) -> dict[str, Any]:
    """아주 단순한 RAG: retrieve -> prompt -> answer

    여기서는 "평가(RAGAS)"가 목적이므로, 구조를 최대한 단순하게 둡니다.
    - Retriever로 컨텍스트(Document들)를 뽑고
    - 그 컨텍스트를 프롬프트에 붙여 답변을 생성합니다.

    Returns:
        dict:
            - question: 원 질문
            - answer: 생성된 답변
            - contexts: retriever가 반환한 컨텍스트 문자열 리스트
    """

    # 1) retrieve: 질문으로 컨텍스트를 가져옵니다.
    docs = list(retriever.invoke(question))
    contexts = [d.page_content for d in docs]

    # 너무 긴 컨텍스트는 평가/출력 가독성을 해치므로 적당히 잘라서 사용
    context_text = "\n\n".join(contexts)
    context_text = context_text[:8000]

    prompt = (
        "너는 시스템 관리자 매뉴얼을 바탕으로 질문에 답하는 도우미야.\n"
        "반드시 아래 [컨텍스트]에 근거해서만 답해.\n\n"
        "[컨텍스트]\n"
        f"{context_text}\n\n"
        "[질문]\n"
        f"{question}\n\n"
        "[답변]"
    )

    # 2) generate: 컨텍스트 기반으로 답변을 생성합니다.
    answer_msg = llm.invoke(prompt)
    answer = getattr(answer_msg, "content", str(answer_msg))

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


def evaluate_with_ragas(samples: list[dict[str, Any]]):
    """RAGAS로 (question, answer, contexts) 샘플들을 평가합니다.

    이 함수가 "RAGAS 튜토리얼"의 핵심입니다.

    우리가 가진 것
    - RAG 파이프라인을 돌려서 만든 샘플들:
        - question: 사용자 질문
        - answer: RAG가 생성한 답변
        - contexts: retriever가 가져온 컨텍스트(문서 chunk들)

    RAGAS가 하는 일
    - 위 샘플들을 입력으로 받아 "메트릭"을 계산해 점수화합니다.
    - 메트릭 중 일부는 LLM/Embeddings를 내부적으로 사용합니다.
        - 그래서 "평가" 자체도 OpenAI API 비용이 발생할 수 있습니다.

    이 튜토리얼에서 쓰는 메트릭(2개)
    - faithfulness(충실성): 답변이 컨텍스트에서 직접 뒷받침되는가?
        - 컨텍스트에 없는 내용을 지어내면(환각) 점수가 떨어지기 쉽습니다.
    - answer_relevancy(답변 관련성): 답변이 질문에 제대로 답했는가?
        - 질문과 동떨어진 답/회피성 답이면 점수가 떨어지기 쉽습니다.

    Returns:
        ragas 평가 결과 객체(dict-like). 보통 print(result) 또는 result.to_pandas()가 가능합니다.
    """

    # 0) 입력 샘플 준비
    # samples는 아래 키를 가진 dict들의 리스트입니다.
    # - question: str
    # - answer: str
    # - contexts: list[str]
    #
    # 여기서 Dataset이 뭐냐?
    # - HuggingFace `datasets` 라이브러리가 제공하는 "테이블형 데이터" 컨테이너입니다.
    # - 엑셀/CSV처럼 "행(row)"이 여러 개 있고, 각 행은 동일한 "열(column)" 구조를 가집니다.
    #   예) 열: question, answer, contexts
    #       행: 질문 1개에 대한 (답변, 컨텍스트) 묶음
    #
    # 왜 굳이 Dataset으로 바꾸냐?
    # - ragas의 `evaluate()`는 내부적으로 이 Dataset 형태를 기준으로
    #   컬럼 이름(user_input/response/retrieved_contexts 등)을 읽고,
    #   배치 처리/비동기 처리/결과 테이블 생성(to_pandas) 등을 편하게 합니다.
    #
    # 참고: Dataset은 "모델"이 아니라 "데이터 묶음"입니다.
    from datasets import Dataset

    # Dataset.from_list는 dict 리스트를 받아서
    # (열=키, 행=각 dict) 형태의 테이블로 만들어줍니다.
    dataset = Dataset.from_list(samples)

    # 1) ragas 평가 함수 및 메트릭 import
    # ragas는 버전에 따라 import 경로가 바뀌고 DeprecationWarning도 자주 나옵니다.
    # 하지만 "이 튜토리얼(현재 환경: ragas 0.4.x)에서 잘 도는 조합"을 고정해둡니다.
    #
    # - evaluate: dataset + metrics + (llm/embeddings)로 평가 실행
    # - faithfulness / answer_relevancy: 튜토리얼에서 사용할 메트릭
    #
    # 참고: 이 import는 DeprecationWarning이 뜰 수 있지만,
    #       지금 환경에선 evaluate()가 기대하는 타입/동작과 잘 맞습니다.
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    # 2) LLM/Embeddings 준비(평가용)
    # RAGAS 메트릭 중 일부는 아래를 사용합니다.
    # - LLM: 답변/컨텍스트를 보고 판정(예: faithfulness)
    # - Embeddings: 텍스트 유사도 계산 등(예: answer_relevancy)
    #
    # 중요: "평가"를 위해서도 LLM/Embeddings가 호출될 수 있어 API 비용이 발생합니다.
    # (RAG 답변 생성 비용 + RAGAS 평가 비용)
    llm = build_llm()  # 평가에 사용할 LLM(여기선 RAG 답변 생성과 같은 모델)

    # answer_relevancy는 embeddings가 필요합니다.
    # - RAGAS_EMBEDDING_MODEL이 없으면 OpenAI 기본 임베딩 모델을 사용합니다.
    from langchain_openai import OpenAIEmbeddings

    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # 3) ragas에 LLM/Embeddings를 넘기는 방법
    # ragas 0.4.x에서는 LangChain 객체를 그대로 받지 않고,
    # 래퍼(wrapper)를 통해서 받는 경우가 있습니다.
    #
    # - LangchainLLMWrapper: LangChain LLM을 ragas가 이해하는 인터페이스로 감쌉니다.
    # - LangchainEmbeddingsWrapper: LangChain Embeddings를 ragas가 이해하는 인터페이스로 감쌉니다.
    #
    # 이 래퍼들은 "deprecated" 경고가 뜰 수 있지만(미래 버전에서 교체 예정),
    # 현재 튜토리얼 환경에서는 동작이 가장 안정적이었습니다.
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # 4) 실제 평가 실행
    # 결과는 전체 평균 점수(예: {'faithfulness': 0.88, 'answer_relevancy': 0.41})와
    # 행별 점수(각 질문 샘플별 점수)를 함께 포함할 수 있습니다.
    # faithfulness(충실성)는 “답변이 검색된 컨텍스트(contexts)에 실제로 근거하고 있나?”를 봅니다.
    #   높은 경우: 답변의 핵심 주장들이 컨텍스트 안에서 직접 뒷받침됨(컨텍스트에 있는 내용 위주로 요약/재진술).
    #   낮은 경우: 컨텍스트에 없는 내용을 추가로 지어내거나(환각), 과하게 일반론/추론으로 답해서 “컨텍스트 기반”이라고 보기 어려움.
    #   해석 팁: 점수가 낮으면 보통 “검색이 엉뚱한 문서를 가져옴 / 컨텍스트가 부족함 / 프롬프트가 ‘컨텍스트만 써라’를 못 지킴 / 모델이 추론으로 메움” 같은 신호예요.
    # answer_relevancy(답변 관련성)는 “답변이 질문에 제대로 대답했나? (핵심을 맞췄나)”를 봅니다.
    #   높은 경우: 질문이 요구한 항목(원인, 절차, 조치, 조건 등)을 직접적으로 다루고, 불필요한 말이 적음.
    #   낮은 경우: 말은 길지만 질문 핵심을 비켜가거나(회피/딴소리), 너무 포괄적으로만 말해서 실제로 질문에 ‘답’이 안 됨.
    #   해석 팁: 이 점수가 낮으면 “검색은 괜찮았는데 프롬프트/답변 스타일이 질문에 맞게 구조화되지 않음”인 경우가 자주 있습니다.
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )

    return result


def main() -> None:
    """샘플을 만들고 RAGAS로 평가합니다.

    튜토리얼 포인트
    - 먼저 "내 RAG가 뭘 뱉는지"(답변/컨텍스트)를 눈으로 확인하고
    - 그 다음 RAGAS 점수로 "어떤 문제"가 있는지 감을 잡습니다.
    """

    _require_openai_key()

    # 실행 제어(튜토리얼용)
    # - 샘플 출력만 보고 싶으면 False로 바꾸면 됩니다.
    run_ragas = True

    # Vector Retriever 준비
    # - tech_manual.md를 임베딩해서 만든 Chroma(VectorDB)를 로드하고
    # - 질문을 임베딩해서 유사한 문서 chunk를 가져옵니다.
    retriever = Utils.load_vector_db().as_retriever(k=6)
    llm = build_llm()

    # 평가용 질문 샘플(적당히 2~3개로 시작)
    questions = [
        "CPU 과부하 상태에서 어떤 조치를 해야 해?",
        "디스크가 가득 찼을 때(용량 부족) 어떤 절차로 해결해?",
        "메모리 누수처럼 보이는데 실제로는 디스크 문제일 수도 있을까?",
    ]

    # (질문 → 컨텍스트 → 답변) 샘플 생성
    samples: list[dict[str, Any]] = []
    for q in questions:
        sample = rag_answer(q, retriever=retriever, llm=llm)
        samples.append(sample)

    print("\n[RAG 답변 샘플]")
    for i, s in enumerate(samples, start=1):
        print(f"\n[{i}] Q: {s['question']}")
        print(f"A: {s['answer'][:400]}" + ("..." if len(s["answer"]) > 400 else ""))
        print(f"contexts: {len(s['contexts'])}개")

    if not run_ragas:
        print("\n[안내] run_ragas=False 이므로 평가를 건너뜁니다.")
        return

    # RAGAS 평가 실행
    # - RAGAS는 메트릭 계산에 LLM/Embeddings를 써서 비용이 추가로 발생할 수 있습니다.
    print("\n[RAGAS 평가 실행] (LLM 호출이 추가로 발생할 수 있습니다)")
    result = evaluate_with_ragas(samples)
    

    # ragas 결과는 보통 dict-like 또는 pandas 변환을 제공합니다.
    try:
        print( f"\n[RAGAS 평가 RAW 결과] -> {result}")
        if hasattr(result, "to_pandas"):
            print("\n[평가 결과 테이블]")
            print(result.to_pandas())
    except Exception:
        print("평가 결과 출력 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main()
