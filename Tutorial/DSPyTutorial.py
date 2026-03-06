"""DSPyTutorial.py

목표
- DSPy의 핵심 개념(Signature, Predict, ChainOfThought, Module, Optimizer)을 단계별로 익힌다.
- 수동으로 프롬프트를 수정하던 작업을 DSPy가 어떻게 자동화하는지 이해한다.
- (실습 3) BootstrapFewShot Optimizer로 few-shot 예시를 자동으로 선택하는 흐름을 체험한다.

예시 도메인 안내
- 이 튜토리얼은 '상품 리뷰 감성 분류'를 예시 도메인으로 사용한다.
- review(리뷰 텍스트), sentiment(감성), reply(답변) 등은 모두 예시용 필드명이며 DSPy 용어가 아니다.
- DSPy 고유 용어: Signature, InputField, OutputField, Predict, ChainOfThought, Module, Optimizer, Example, Prediction

실행 준비
1) 프로젝트 루트의 .env 파일에 키가 있어야 합니다.
   - OPENAI_API_KEY=... (필수)
   - (선택) OPENAI_MODEL=gpt-4o-mini  또는 원하는 모델명

2) 실행
     - Windows PowerShell:
         ./.venv/Scripts/python.exe ./Tutorial/DSPyTutorial.py

핵심 개념 요약
- Signature  : 입력/출력 필드를 선언하는 "계약". 프롬프트 대신 씀.
- Predict    : Signature를 받아 LLM을 한 번 호출하는 가장 기본 모듈.
- ChainOfThought: Predict에 추론 과정(rationale) 자동 추가.
- Module     : 여러 Predict/ChainOfThought를 조합한 파이프라인. PyTorch nn.Module과 유사.
- Optimizer  : (예: BootstrapFewShot) 평가 데이터 + 지표를 받아 few-shot 예시를 자동 선택.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import dspy
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 공통: 프로젝트 루트 기준으로 동작하게 고정
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 공통: 환경변수 로드 + DSPy LM 설정
# ---------------------------------------------------------------------------

def configure_dspy() -> None:
    """프로젝트 루트의 .env를 읽어 DSPy LM을 설정합니다."""

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # dspy.LM: DSPy가 LLM을 호출할 때 사용하는 래퍼 객체.
    # model 형식은 "provider/model명" 규칙을 따른다 (내부적으로 LiteLLM을 사용).
    # 예) "openai/gpt-4o-mini", "anthropic/claude-3-5-haiku-20241022" 등
    lm = dspy.LM(model=f"openai/{model}", api_key=api_key)

    # dspy.configure(lm=...): 이 프로세스 전체에서 사용할 기본 LM을 전역으로 등록한다.
    # 이후 dspy.Predict / dspy.ChainOfThought 등 모든 모듈은
    # 별도로 LM을 지정하지 않아도 여기서 등록한 lm을 자동으로 사용한다.
    # → LangChain의 llm 객체를 각 체인마다 직접 넘기던 방식과 달리,
    #   "한 번 설정하면 어디서든 자동 적용"되는 전역 설정 개념이다.
    dspy.configure(lm=lm)
    print(f"[DSPy 설정 완료] 모델: {model}")


# ===========================================================================
# 튜토리얼 1 - Signature + Predict (가장 기본)
# ===========================================================================
#
# [학습 포인트]
# - Signature 클래스로 입출력을 선언하면 DSPy가 프롬프트를 자동 생성한다.
# - Predict(Signature)는 LLM을 한 번 호출하는 가장 단순한 모듈이다.
# - result.필드명 처럼 OutputField 이름으로 출력값에 바로 접근 가능하다.
# ===========================================================================


# [DSPy 용어] Signature: 입력/출력 필드를 선언하는 클래스.
# 클래스 docstring이 LLM에게 전달되는 태스크 설명(지시문)이 된다.
# InputField / OutputField 는 각 필드의 역할을 LLM에게 알려주는 메타데이터다.
class ReviewClassifySignature(dspy.Signature):

    # LLM이 이 docstring을 읽고 "상품 리뷰 감성 분류" 태스크로 이해하도록 지시문 역할을 합니다.
    """상품 리뷰를 읽고 감성(sentiment)을 분류합니다."""

    # review, sentiment, reason 은 이 예시에서 사용하는 필드명일 뿐, DSPy 예약어가 아니다.
    # 어떤 이름이든 자유롭게 지을 수 있다.
    review: str = dspy.InputField(desc="분류할 상품 리뷰 텍스트")
    sentiment: str = dspy.OutputField(desc="감성: positive / negative / neutral 중 하나")
    reason: str = dspy.OutputField(desc="분류 근거를 1~2문장으로 설명")


def tutorial_basic_predict() -> None:
    """
        (튜토리얼 1) Signature + Predict로 상품 리뷰 감성 분류.
        메세지 클래스를(Signature) 정의 하고, 그걸로 Predict 모듈을 만들어서 LLM을 호출하는 가장 기본적인 흐름입니다.
        사용자 입력 메세지를 Predict 모듈에 넣으면, Signature에 정의된 InputField/OutputField 정보를 바탕으로 
        DSPy가 자동으로 프롬프트를 생성해서 LLM을 호출합니다.
        중요한건, 이 과정에서 프롬프트를 사용자가 직접 입력한건 하나도 없다는 것.
    """

    configure_dspy()

    # [DSPy 용어] Predict: Signature를 받아 LLM을 한 번 호출하는 가장 기본 모듈.
    # DSPy가 Signature의 docstring + 필드 정보를 조합해 프롬프트를 자동 생성한다.
    # 사람이 직접 프롬프트 문자열을 작성할 필요가 없다.
    # Predict는 Few-shot 예시나 추론 과정 없이 "즉시 답변"이 필요한 간단한 태스크에 적합하다.
    classify = dspy.Predict(ReviewClassifySignature)

    print("\n[튜토리얼 1] Signature + Predict")
    print("상품 리뷰를 입력하면 감성(positive/negative/neutral)을 분류합니다.")
    print("[예시] '배송이 너무 빠르고 포장도 꼼꼼해요!' / '품질이 사진과 너무 달라요'\n")

    while True:
        review_text = input("리뷰 텍스트 (Enter로 종료): ").strip()
        if not review_text:
            break

        # review 는 ReviewClassifySignature에 선언한 InputField 이름과 일치해야 한다.
        # 내부적으로 dspy.Module의 forward() 메서드를 호출 하는데, 이렇게 객체를 메서드 처럼 쓸 수 있는 이유는
        # 파이썬의 __call__ 메서드(같은 역거운 장난질) 덕분이며, 여기서 forward()를 호출함.
        # 리턴타입은 Prediction.
        result: dspy.Prediction = classify(review=review_text)

        # Prediction 객체가 Signature의 OutputField 이름을 그대로 속성으로 제공한다.
        # 어떻게? ->**kwargs 문법과 __getattr__ 문법 덕분에. (또 파이썬의 문법 장난질)
        print(f"\n  감성 : {result.sentiment}")
        print(f"  근거 : {result.reason}\n")


# ===========================================================================
# 튜토리얼 2 - ChainOfThought + Module (파이프라인 구성)
# ===========================================================================
#
# [학습 포인트]
# - ChainOfThought는 Predict와 동일하지만 "추론 과정(rationale)"을 자동으로 생성한다.
#   → 복잡한 분석·판단 작업에서 정확도가 높아진다.
# - Module로 "분류 → 조치 권고" 2단계 파이프라인을 하나의 클래스로 묶는다.
#   PyTorch의 nn.Module과 동일한 패턴이라 .parameters(), .save() 등이 동일하게 동작한다.
# ===========================================================================


class ReplyDraftSignature(dspy.Signature):
    """리뷰 감성을 참고해 고객에게 보낼 답변 초안을 작성합니다."""

    review: str = dspy.InputField(desc="원본 리뷰 텍스트")
    sentiment: str = dspy.InputField(desc="감성 분류 결과 (positive/negative/neutral)")
    reply: str = dspy.OutputField(desc="고객에게 보낼 답변 초안 (2~3문장)")


# [DSPy 용어] Module: 여러 Predict/ChainOfThought를 조합하는 파이프라인 클래스.
# PyTorch의 nn.Module과 동일한 패턴 — __init__에 서브모듈 등록, forward에 로직 작성.
class ReviewAnalysisPipeline(dspy.Module):
    """리뷰 감성 분류 → 답변 초안 작성 2단계 파이프라인."""

    def __init__(self) -> None:
        super().__init__()
        # [DSPy 용어] ChainOfThought: Predict와 동일하나 LLM이 답변 전에
        # 추론 과정(rationale)을 먼저 생성한다. 복잡한 판단 태스크에서 정확도가 높아진다.
        # 다만 CoT만 추가 되었을 뿐, Few-shot은 여전히 없다.
        self.classify = dspy.ChainOfThought(ReviewClassifySignature)
        self.draft_reply = dspy.ChainOfThought(ReplyDraftSignature)

    def forward(self, review: str) -> dspy.Prediction:
        # 1단계: 감성 분류
        classified = self.classify(review=review)

        # 2단계: 분류 결과를 다음 모듈의 입력으로 연결
        drafted = self.draft_reply(
            review=review,
            sentiment=classified.sentiment,
        )

        # [DSPy 용어] Prediction: 여러 출력 필드를 담는 결과 객체.
        return dspy.Prediction(
            sentiment=classified.sentiment,
            reason=classified.reason,
            reply=drafted.reply,
        )


def tutorial_cot_module() -> None:
    """(튜토리얼 2) ChainOfThought + Module로 감성 분류→답변 초안 파이프라인."""

    configure_dspy()

    pipeline = ReviewAnalysisPipeline()

    print("\n[튜토리얼 2] ChainOfThought + Module (2단계 파이프라인)")
    print("리뷰 → 감성 분류 → 고객 답변 초안 작성 순서로 처리합니다.")
    print("[예시] '색상이 예쁘고 착용감도 좋아요' / '사이즈가 너무 작게 나왔어요'\n")

    while True:
        review_text = input("리뷰 텍스트 (Enter로 종료): ").strip()
        if not review_text:
            break

        # 메서드를 명시하고 않고 이렇게 호출하면 내부적으로 __call__ 메서드를 호출하며,
        # DSPy는 __call__() 내부에서 forward()를 호출, forward()에서 정의한 로직에 따라 classify → draft_reply 순서로 실행한다.
        result = pipeline(review=review_text)

        print(f"\n  감성   : {result.sentiment}")
        print(f"  근거   : {result.reason}")
        print(f"\n  답변 초안:")
        print(f"  {result.reply}\n")


# ===========================================================================
# 튜토리얼 3 - BootstrapFewShot Optimizer (자동 프롬프트 최적화)
# ===========================================================================
#
# [학습 포인트]
# - DSPy Optimizer의 핵심: 사람이 few-shot 예시를 직접 고르던 작업을 자동화.
# - BootstrapFewShot: trainset에서 LLM이 직접 정답을 생성하고,
#   metric 함수로 평가해 "좋은 예시"만 자동으로 선택한다.
# - compile() 후 반환된 모듈은 최적화된 프롬프트(few-shot 포함)를 내장한다.
#
# [흐름 요약]
#   trainset 준비 → metric 정의 → optimizer.compile() → 최적화된 모듈 사용
# ===========================================================================


# [DSPy 용어] Example: (입력, 정답) 쌍을 담는 데이터 객체.
# with_inputs("필드명")으로 어떤 필드가 '입력'인지 DSPy에게 알려준다.
# (지정하지 않은 나머지 필드는 '정답 레이블'로 취급된다.)
#
# 아래는 튜토리얼용 소규모 학습 데이터. 실제 최적화에는 수십~수백 개가 필요하다.
TRAIN_EXAMPLES = [
    dspy.Example(
        review="배송이 정말 빠르고 포장도 꼼꼼했어요. 완전 만족합니다!",
        sentiment="positive",
    ).with_inputs("review"),
    dspy.Example(
        review="색상이 사진과 너무 달라요. 많이 실망했습니다.",
        sentiment="negative",
    ).with_inputs("review"),
    dspy.Example(
        review="가격 대비 나쁘지 않은 것 같아요. 보통이에요.",
        sentiment="neutral",
    ).with_inputs("review"),
    dspy.Example(
        review="소재가 부드럽고 착용감이 편안해서 자주 입게 될 것 같아요.",
        sentiment="positive",
    ).with_inputs("review"),
    dspy.Example(
        review="사이즈가 너무 작게 나왔고 반품도 번거로웠어요.",
        sentiment="negative",
    ).with_inputs("review"),
    dspy.Example(
        review="딱히 특별한 점은 없지만 그렇다고 나쁘지도 않아요.",
        sentiment="neutral",
    ).with_inputs("review"),
]


def sentiment_accuracy_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """
    [DSPy 용어] metric 함수: Optimizer가 "좋은 예시"를 판별하는 기준.
    True를 반환하면 해당 예시를 few-shot 후보로 채택한다.

    - example.sentiment : TRAIN_EXAMPLES에 적은 정답 레이블
    - pred.sentiment    : 모델이 예측한 값 (OutputField)
    - trace             : Optimizer 내부 추론 추적 객체 (평가 모드에서는 None)
    """
    
    return example.sentiment.strip().lower() == pred.sentiment.strip().lower()


def tutorial_optimizer() -> None:
    """(튜토리얼 3) BootstrapFewShot Optimizer로 자동 최적화."""

    configure_dspy()

    # -----------------------------------------------------------------------
    # 최적화 전: 기본 모듈 (few-shot 없이 Signature만으로 동작)
    # -----------------------------------------------------------------------
    base_classifier = dspy.Predict(ReviewClassifySignature)

    print("\n[튜토리얼 3] BootstrapFewShot Optimizer")
    print("=" * 60)
    print("1단계: 최적화 전 기본 모듈로 테스트 (few-shot 없음)")

    test_input = "디자인은 예쁜데 한 달 쓰니까 박음질이 뜯어졌어요."
    result_before = base_classifier(review=test_input)
    print(f"\n  입력   : {test_input}")
    print(f"  감성   : {result_before.sentiment}")
    print(f"  근거   : {result_before.reason}")

    # -----------------------------------------------------------------------
    # 최적화: BootstrapFewShot 실행
    # -----------------------------------------------------------------------
    print("\n2단계: BootstrapFewShot 최적화 실행 중...")
    print("  TRAIN_EXAMPLES를 이용해 LLM이 직접 few-shot 후보를 생성하고,")
    print("  sentiment_accuracy_metric으로 평가해 좋은 예시만 자동 선택합니다.\n")

    # [DSPy 용어] BootstrapFewShot: Optimizer의 한 종류.
    # max_bootstrapped_demos: LLM이 새로 생성하는 few-shot 후보 최대 수
    # max_labeled_demos     : trainset의 원본 레이블을 그대로 쓰는 few-shot 최대 수
    # BootstrapFewShot의 내부 동작 흐름:
    #
    # 1. metric=sentiment_accuracy_metric
    #    → 함수 자체를 객체로 전달 (호출하지 않음, Python에서 함수는 일급 객체)
    #    → Optimizer가 내부적으로 필요한 시점에 직접 호출함
    #
    # 2. compile() 호출 시 Optimizer가 하는 일:
    #    for each example in TRAIN_EXAMPLES:
    #        pred = student(review=example.review)      # LLM으로 예측
    #        ok   = metric(example, pred)               # 정답과 비교
    #        if ok: few_shot_candidates.append(example) # 정답이면 후보로 추가
    #
    # 3. 후보 중에서 최대 max_bootstrapped_demos(3)개를 자동 선택해
    #    student 모듈의 프롬프트 앞에 few-shot 예시로 삽입
    #
    # 4. max_labeled_demos(2): trainset의 원본 (입력, 정답레이블) 쌍을
    #    LLM 예측 없이 그대로 few-shot으로 추가 사용하는 최대 수
    optimizer = dspy.BootstrapFewShot(
        metric=sentiment_accuracy_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
    )

    # [DSPy 용어] compile(): Optimizer가 student를 최적화해 새 모듈을 반환한다.
    #
    # - student=dspy.Predict(ReviewClassifySignature)
    #     최적화 대상 모듈. 아직 few-shot이 없는 "백지 상태"의 Predict 모듈을 넘긴다.
    #     Optimizer는 이 모듈을 복사한 뒤, 내부에 few-shot 예시를 주입해서 반환한다.
    #     (원본 student 객체는 변경되지 않는다)
    #
    # - trainset=TRAIN_EXAMPLES
    #     Optimizer가 few-shot 후보를 탐색할 때 사용할 학습 데이터.
    #     각 Example에 대해 student를 호출(LLM 실행)하고,
    #     metric으로 평가해서 정답이면 few-shot 후보로 채택한다.
    #
    # - 반환값 optimized_classifier
    #     student와 동일한 타입(Predict)이지만,
    #     프롬프트 내부에 자동 선택된 few-shot 예시가 삽입된 상태.
    #     이후 optimized_classifier(review=...) 로 호출하면
    #     few-shot이 포함된 프롬프트로 LLM을 호출한다.
    optimized_classifier = optimizer.compile(
        student=dspy.Predict(ReviewClassifySignature),
        trainset=TRAIN_EXAMPLES,
    )

    # -----------------------------------------------------------------------
    # 최적화 후: 동일 입력으로 비교
    # -----------------------------------------------------------------------
    print("3단계: 최적화 후 모듈로 동일 입력 재테스트 (few-shot 자동 포함)")

    result_after = optimized_classifier(review=test_input)
    print(f"\n  입력   : {test_input}")
    print(f"  감성   : {result_after.sentiment}")
    print(f"  근거   : {result_after.reason}")

    print("\n" + "=" * 60)
    print("[비교 요약]")
    print(f"  최적화 전 감성: {result_before.sentiment}")
    print(f"  최적화 후 감성: {result_after.sentiment}")
    print("\n최적화된 모듈 내부에는 자동 선택된 few-shot 예시가 포함되어 있습니다.")
    print("실제 서비스에서는 더 많은 trainset + 엄밀한 metric으로 품질을 높일 수 있습니다.")

    # -----------------------------------------------------------------------
    # 최적화된 모듈로 직접 입력 테스트
    # -----------------------------------------------------------------------
    print("\n4단계: 최적화된 모듈로 직접 테스트")
    print("[예시] '정말 만족해요!' / '완전 별로예요' / '그냥 그래요'\n")

    while True:
        review_text = input("리뷰 텍스트 (Enter로 종료): ").strip()
        if not review_text:
            break

        result = optimized_classifier(review=review_text)
        print(f"\n  감성 : {result.sentiment}")
        print(f"  근거 : {result.reason}\n")


# ===========================================================================
# 튜토리얼 4 - BootstrapFewShot (정답 없이 품질 기반 최적화)
# ===========================================================================
#
# [학습 포인트]
# - trainset에 정답 레이블이 없어도 BootstrapFewShot을 쓸 수 있다.
# - metric 함수가 "정답과 비교" 대신 "출력 품질 검사"를 하면 된다.
# - 정답을 모르는 실무 상황(RAG 쿼리 생성, 요약 등)에서 유용한 패턴이다.
#
# [튜토리얼 3과의 차이]
#   튜토리얼 3: Example에 sentiment(정답) 포함 → metric이 정답과 비교
#   튜토리얼 4: Example에 입력만 포함       → metric이 출력 형식/품질만 검사
# ===========================================================================


# 입력(review)만 있고 정답 레이블(sentiment)이 없는 trainset.
# with_inputs("review")로 review가 입력 필드임을 DSPy에 알린다.
# sentiment, reason 필드를 제공하지 않으면 DSPy는 이를 정답 없는 예시로 취급한다.
TRAIN_EXAMPLES_NO_LABEL = [
    dspy.Example(review="배송이 정말 빠르고 포장도 꼼꼼했어요.").with_inputs("review"),
    dspy.Example(review="색상이 사진과 너무 달라서 실망했어요.").with_inputs("review"),
    dspy.Example(review="가격 대비 나쁘지 않은 것 같아요.").with_inputs("review"),
    dspy.Example(review="소재가 부드럽고 착용감이 편안해요.").with_inputs("review"),
    dspy.Example(review="사이즈가 너무 작게 나왔어요.").with_inputs("review"),
    dspy.Example(review="딱히 특별한 점은 없지만 나쁘지도 않아요.").with_inputs("review"),
]


def quality_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """
    정답 없이 출력 품질만으로 판단하는 metric.

    example.sentiment 같은 정답 레이블이 없으므로 비교하지 않는다.
    대신 출력이 "쓸만한가"를 형식/내용 기준으로만 검사한다.

    - pred.sentiment : OutputField로 선언된 출력값
    - pred.reason    : OutputField로 선언된 출력값
    """
    sentiment = pred.sentiment.strip().lower()
    reason = pred.reason.strip()

    # 1. sentiment가 허용된 값 중 하나인지 확인
    if sentiment not in ("positive", "negative", "neutral"):
        return False

    # 2. reason이 너무 짧으면 제외 (성의 없는 출력)
    if len(reason) < 10:
        return False

    # 3. reason에 sentiment 키워드가 아예 없으면 제외 (근거와 결론이 무관한 경우)
    sentiment_keywords = {
        "positive": ["좋", "만족", "빠르", "편안", "추천", "훌륭"],
        "negative": ["아쉽", "실망", "별로", "불편", "문제", "나쁘"],
        "neutral":  ["보통", "그냥", "평범", "나쁘지", "특별"],
    }
    keywords = sentiment_keywords.get(sentiment, [])
    if keywords and not any(kw in reason for kw in keywords):
        return False

    return True


def tutorial_optimizer_no_label() -> None:
    """(튜토리얼 4) 정답 없이 품질 기반 metric으로 BootstrapFewShot 최적화."""

    configure_dspy()

    print("\n[튜토리얼 4] BootstrapFewShot - 정답 없는 품질 기반 최적화")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 최적화 전 테스트
    # -----------------------------------------------------------------------
    base_classifier = dspy.Predict(ReviewClassifySignature)

    print("1단계: 최적화 전 기본 모듈로 테스트 (few-shot 없음)")
    test_input = "디자인은 예쁜데 한 달 쓰니까 박음질이 뜯어졌어요."
    result_before = base_classifier(review=test_input)
    print(f"\n  입력   : {test_input}")
    print(f"  감성   : {result_before.sentiment}")
    print(f"  근거   : {result_before.reason}")

    # -----------------------------------------------------------------------
    # 최적화 실행
    # -----------------------------------------------------------------------
    print("\n2단계: BootstrapFewShot 최적화 실행 중 (정답 없는 trainset + 품질 metric)...")
    print("  TRAIN_EXAMPLES_NO_LABEL: sentiment 레이블 없이 review 입력만 포함.")
    print("  quality_metric: 정답 비교 없이 출력 형식/품질만 검사.\n")

    # 3번 튜토리얼과 완전히 동일한 Optimizer를 쓴다.
    # 달라지는 건 trainset(정답 없음)과 metric(품질 검사) 뿐이다.
    #
    # Optimizer 내부 동작 (정답 없는 경우):
    #   for each example in TRAIN_EXAMPLES_NO_LABEL:
    #       pred = student(review=example.review)   # LLM으로 예측
    #       ok   = quality_metric(example, pred)    # 정답 비교 X, 품질만 검사
    #       if ok: few_shot_candidates.append(...)  # 품질 통과 시 few-shot 후보로 채택
    optimizer = dspy.BootstrapFewShot(
        metric=quality_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=0,  # 정답 레이블이 없으므로 0으로 설정
    )

    optimized_classifier = optimizer.compile(
        student=dspy.Predict(ReviewClassifySignature),
        trainset=TRAIN_EXAMPLES_NO_LABEL,
    )

    # -----------------------------------------------------------------------
    # 최적화 후 비교
    # -----------------------------------------------------------------------
    print("3단계: 최적화 후 모듈로 동일 입력 재테스트")
    result_after = optimized_classifier(review=test_input)
    print(f"\n  입력   : {test_input}")
    print(f"  감성   : {result_after.sentiment}")
    print(f"  근거   : {result_after.reason}")

    print("\n" + "=" * 60)
    print("[비교 요약]")
    print(f"  최적화 전 감성: {result_before.sentiment}")
    print(f"  최적화 후 감성: {result_after.sentiment}")
    print("\n정답 레이블 없이도 품질 기반 metric으로 few-shot 예시를 자동 선택했습니다.")
    print("RAG 쿼리 생성, 요약, 재작성 등 정답을 정의하기 어려운 태스크에 이 패턴을 사용하세요.")

    # -----------------------------------------------------------------------
    # 직접 테스트
    # -----------------------------------------------------------------------
    print("\n4단계: 최적화된 모듈로 직접 테스트")
    print("[예시] '정말 만족해요!' / '완전 별로예요' / '그냥 그래요'\n")

    while True:
        review_text = input("리뷰 텍스트 (Enter로 종료): ").strip()
        if not review_text:
            break

        result = optimized_classifier(review=review_text)
        print(f"\n  감성 : {result.sentiment}")
        print(f"  근거 : {result.reason}\n")


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    print("=" * 60)
    print("DSPy 튜토리얼")
    print("=" * 60)
    print("1) Signature + Predict      (기본: 선언형 프롬프트 / 리뷰 감성 분류)")
    print("2) ChainOfThought + Module  (중급: 추론 + 파이프라인 / 분류→답변 초안)")
    print("3) BootstrapFewShot         (고급: 자동 최적화 / 정답 있는 trainset)")
    print("4) BootstrapFewShot         (고급: 자동 최적화 / 정답 없는 품질 기반 metric)")

    choice = input("\n선택 (1/2/3/4): ").strip()

    tutorials: dict[str, Callable[[], None]] = {
        "1": tutorial_basic_predict,
        "2": tutorial_cot_module,
        "3": tutorial_optimizer,
        "4": tutorial_optimizer_no_label,
    }

    func = tutorials.get(choice)
    if func:
        func()
    else:
        print("1, 2, 3, 4 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()
