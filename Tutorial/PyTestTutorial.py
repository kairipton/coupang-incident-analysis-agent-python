"""
pytest 튜토리얼
===============
목표: pytest와 unittest.mock을 사용하여 자동화된 테스트를 작성하는 방법을 익힌다.

실행 방법:
    # 이 파일만 실행
    pytest Tutorial/PyTestTutorial.py -v

    # tests/ 폴더 전체 실행
    pytest tests/ -v

pytest 동작 원리:
    - pytest는 파일을 스캔해서 "test_" 로 시작하는 함수를 모두 찾아 자동 실행한다.
    - 함수 안에서 assert 가 전부 통과하면 → PASSED
    - assert 가 하나라도 실패하면 → FAILED (어디서 왜 실패했는지 출력)
    - 예외가 발생해서 함수가 비정상 종료되면 → ERROR
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage


# ===========================================================================
# 1. 가장 기본적인 테스트
#
# [테스트의 목적]
# 내가 만든 코드가 "기대한 결과"를 내는지 자동으로 확인하는 것.
# assert A == B 는 "A 와 B 가 같아야 한다"는 선언이다.
# 같으면 아무 일도 없고(통과), 다르면 AssertionError 를 발생시켜 테스트를 실패시킨다.
#
# [PASSED 조건]
# assert 문이 전부 참(True)이면 통과.
# ===========================================================================

def test_basic_assert():
    # 1 + 1 은 2 여야 한다. 다르면 FAILED.
    assert 1 + 1 == 2

    # "hello".upper() 는 "HELLO" 여야 한다.
    assert "hello".upper() == "HELLO"

    # 리스트 첫 번째 원소는 1 이어야 한다.
    assert [1, 2, 3][0] == 1


def test_assert_with_message():
    score = 0.85

    # assert 조건, "실패 메시지" 형태로 쓰면,
    # 조건이 거짓일 때 메시지를 출력해줘서 왜 실패했는지 바로 알 수 있다.
    # score 가 0.7 미만이면 FAILED + "점수가 너무 낮음: 0.xx" 출력.
    assert score >= 0.7, f"점수가 너무 낮음: {score}"


# ===========================================================================
# 2. 예외 발생 테스트
#
# [언제 쓰나]
# "이 잘못된 입력을 넣으면 반드시 예외가 발생해야 한다"를 검증할 때.
# 예) 음수 입력 시 ValueError, 없는 키 접근 시 KeyError 등.
#
# [중요: 헷갈리지 말 것]
# pytest.raises(예외타입) 블록 안에서 해당 예외가 "발생하면" → PASSED
# 예외가 발생하지 않으면 → FAILED
# 즉, "예외가 터지는 게 정상인 상황"을 테스트하는 것이다.
#
# [1번과의 차이]
# 1번: 정상 동작 결과를 검증 (assert 값 == 기대값)
# 2번: 비정상 입력에 대한 방어 코드가 올바르게 예외를 던지는지 검증
# ===========================================================================

def test_expect_exception():
    import pytest

    # 1 / 0 은 ZeroDivisionError 를 발생시킨다.
    # pytest.raises(ZeroDivisionError) 는 "이 블록 안에서 ZeroDivisionError 가 나야 통과"라는 의미.
    # 예외가 발생했으므로 → PASSED
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0

    # 없는 키에 접근하면 KeyError 가 발생한다.
    # 역시 예외가 발생했으므로 → PASSED
    with pytest.raises(KeyError):
        d = {}
        _ = d["없는키"]


# ===========================================================================
# 3. mock 기본 - 외부 의존성을 가짜로 대체
#
# [문제 상황]
# 테스트하려는 함수가 내부에서 OpenAI API 를 호출한다면?
# → 테스트할 때마다 실제 API 비용이 발생하고, 네트워크 상태에 따라 결과가 달라진다.
#
# [해결책: Mock]
# MagicMock() 으로 "가짜 객체"를 만들어서 실제 API 대신 주입한다.
# 가짜 객체는 내가 원하는 값을 반환하도록 설정할 수 있다.
# 이렇게 하면 네트워크/API 없이 함수의 "로직"만 순수하게 검증할 수 있다.
# ===========================================================================

def 내가_만든_함수(llm):
    """테스트 대상 함수 - llm 을 받아서 invoke 하고 content 를 반환"""
    response = llm.invoke("질문")
    return response.content


def test_mock_basic():
    # MagicMock() 은 어떤 메서드를 호출해도 에러가 안 나는 "만능 가짜 객체"다.
    fake_llm = MagicMock()
    
    # fake_llm.invoke() 를 호출하면 AIMessage(content="가짜 답변") 을 반환하도록 설정.
    # 실제 OpenAI API 는 전혀 호출되지 않는다.
    fake_llm.invoke.return_value = AIMessage(content="가짜 답변")

    # 테스트 대상 함수에 가짜 llm 을 주입해서 실행
    result = 내가_만든_함수(fake_llm)

    # 반환값이 "가짜 답변" 이어야 한다.
    assert result == "가짜 답변"

    # fake_llm.invoke 가 "질문" 이라는 인자로 정확히 1번 호출됐는지도 검증 가능.
    # 함수 내부에서 llm.invoke 를 빠뜨리거나, 다른 인자로 호출했다면 FAILED.
    fake_llm.invoke.assert_called_once_with("질문")


# ===========================================================================
# 4. patch 데코레이터 - 모듈 안에 이미 import 된 객체를 가짜로 교체
#
# [3번과의 차이]
# 3번은 함수 인자로 가짜 객체를 직접 전달했다.
# 4번은 모듈 내부에 이미 존재하는 변수/함수를 테스트 중에만 몰래 교체한다.
#
# [patch 경로 규칙]
# patch("패키지.모듈.교체할대상") 형태로 지정한다.
# 예) AI.Node 모듈 안의 llm 변수를 교체하려면 → patch("AI.Node.llm")
# ===========================================================================

# 테스트 대상 함수들 (실제 프로젝트의 Node.py 함수와 같은 구조)
def get_current_time():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d")  # 실행할 때마다 오늘 날짜가 바뀜


def 오늘_날짜_포함된_메시지():
    # 내부에서 get_current_time() 을 호출함
    return f"오늘은 {get_current_time()} 입니다."


# @patch(경로) 데코레이터를 붙이면,
# 테스트 함수가 실행되는 동안만 get_current_time 을 가짜로 교체해준다.
# 교체된 가짜 객체(mock_time)는 함수 인자로 자동 주입된다.
@patch(__name__ + ".get_current_time")  # __name__ = 이 파일의 모듈 경로
def test_patch_decorator(mock_time):
    # get_current_time() 이 호출되면 "2030-01-01" 을 반환하도록 설정
    mock_time.return_value = "2030-01-01"

    result = 오늘_날짜_포함된_메시지()

    # 실제 오늘 날짜가 아니라 "2030-01-01" 이 들어간 결과여야 한다.
    assert result == "오늘은 2030-01-01 입니다."
    # 테스트 함수가 끝나면 get_current_time 은 원래 함수로 자동 복원된다.


# ===========================================================================
# 5. patch context manager - with 블록으로 교체 범위를 직접 제어
#
# [4번과의 차이]
# 데코레이터는 함수 전체에 적용된다.
# with patch(...) 는 블록 안에서만 적용되고, 블록을 나오면 즉시 복원된다.
# 한 테스트 안에서 "여기서만 교체하고, 저기서는 원래 값으로 돌아와야 한다"
# 는 상황에 적합하다.
# ===========================================================================

def test_patch_context_manager():
    # with 블록 안에서만 get_current_time 이 가짜로 교체됨
    with patch(__name__ + ".get_current_time") as mock_time:
        mock_time.return_value = "2099-12-31"
        result = 오늘_날짜_포함된_메시지()
        assert "2099-12-31" in result  # 블록 안: 가짜 날짜가 들어있어야 함

    # with 블록을 벗어나면 get_current_time 은 원래 함수로 복원된다.
    # 따라서 result_real 에는 실제 오늘 날짜가 들어있고, "2099-12-31" 은 없어야 한다.
    result_real = 오늘_날짜_포함된_메시지()
    assert "2099-12-31" not in result_real


# ===========================================================================
# 6. 실제 프로젝트 노드 함수 테스트 패턴
#
# [목적]
# node_final_answer 는 내부에서 llm.stream() 을 호출한다.
# 실제 API 를 호출하지 않고, 노드 함수의 "로직"만 검증한다.
#   - state 입력을 받아서
#   - messages 에 AIMessage 를 추가해서 반환하는가?
#
# [patch 경로]
# Node.py 안에서 llm 은 모듈 레벨 변수로 선언되어 있다.
# → patch("AI.Node.llm") 으로 교체한다.
# ===========================================================================

def test_node_final_answer_pattern():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with patch("AI.Node.llm") as mock_llm:
        # llm.stream() 은 청크(조각) 리스트를 반환한다.
        # 가짜 청크를 만들어서 content 를 "쿠팡 사고 요약입니다." 로 설정.
        fake_chunk = MagicMock()
        fake_chunk.content = "쿠팡 사고 요약입니다."
        mock_llm.stream.return_value = [fake_chunk]  # 청크 1개짜리 리스트 반환

        import AI.Node as Node

        # 노드 함수에 넘길 가짜 State
        state = {
            "question": "쿠팡 사고 원인은?",
            "messages": [HumanMessage(content="쿠팡 사고 원인은?")],
            "documents": ["관련 문서 내용"],
            "summary": ""
        }

        result = Node.node_final_answer(state)

        # 검증 1: 반환값에 "messages" 키가 있어야 한다.
        assert "messages" in result

        # 검증 2: messages 에 정확히 1개의 메시지가 추가됐어야 한다.
        assert len(result["messages"]) == 1

        # 검증 3: 추가된 메시지의 content 가 가짜 청크의 content 와 같아야 한다.
        assert result["messages"][0].content == "쿠팡 사고 요약입니다."


# ===========================================================================
# 7. lru_cache 캐싱 동작 검증
#
# [목적]
# lru_cache 가 붙은 함수는 같은 인자로 호출 시 내부 로직을 실행하지 않고
# 첫 번째 결과를 그대로 돌려준다.
# 이 테스트는 캐싱이 실제로 동작하는지 검증한다.
#
# [is 연산자]
# == 는 값이 같은지 비교, is 는 메모리상 완전히 동일한 객체인지 비교.
# 캐싱이 동작하면 r1, r2, r3 는 같은 메모리 주소를 가리켜야 한다.
# ===========================================================================

def test_lru_cache_pattern():
    from functools import lru_cache

    call_count = 0  # 함수가 실제로 몇 번 실행됐는지 카운트

    @lru_cache(maxsize=1)
    def 비싼_함수(k: int):
        nonlocal call_count
        call_count += 1       # 실제 실행될 때마다 +1
        return object()       # 호출할 때마다 새 객체를 만들려고 하지만, 캐싱되면 안 만들어짐

    r1 = 비싼_함수(5)  # 최초 호출 → 실제 실행, call_count = 1
    r2 = 비싼_함수(5)  # 캐시 히트 → 실제 실행 안 함, call_count 그대로
    r3 = 비싼_함수(5)  # 캐시 히트 → 실제 실행 안 함, call_count 그대로

    # r1, r2, r3 는 모두 동일한 객체여야 한다 (캐싱된 첫 번째 결과).
    assert r1 is r2 is r3

    # 함수 내부 로직은 딱 1번만 실행됐어야 한다.
    assert call_count == 1