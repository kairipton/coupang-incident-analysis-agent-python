"""GraphToolCallingTutorial.py

LangGraph로 Tool Calling(함수 호출)을 구현하는 초보자용 튜토리얼입니다.

핵심 목표
- LLM이 필요할 때 `add(a, b)` 툴을 "자율적으로" 호출하게 만들기
- 툴이 반환한 값을 다시 LLM이 받아서 최종 답변에 반영하는 흐름 이해하기

준비물
- 프로젝트 루트의 .env에 OPENAI_API_KEY 필요
- (선택) OPENAI_MODEL 설정 (없으면 GameConfig.llm_model_name 사용)

실행
- Windows PowerShell (프로젝트 루트에서):
  ./.venv/Scripts/python.exe ./Tutorial/GraphToolCallingTutorial.py

이 튜토리얼에서 배우는 것
1) tool 함수(add)를 @tool로 등록하면 무엇이 달라지는지
2) LLM(ChatOpenAI)에 tools를 bind 하면 어떤 일이 생기는지
3) LangGraph에서 "assistant 노드" ↔ "tools 노드"를 루프로 연결하면
   도구 호출이 자동으로 수행되는 구조가 어떻게 만들어지는지
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict, Any

from dotenv import load_dotenv

# LangChain(LLM/메시지/툴)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

# LangGraph(그래프 + 미리 만들어진 ToolNode)
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode


# ---------------------------------------------------------------------------
# 0) 프로젝트 루트 고정 (import/.env 경로 문제를 피하기 위해)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Tutorial 폴더에서 실행하면 기본 sys.path에 루트가 없을 수 있어
    # GameConfig 같은 루트 모듈을 import 못 하는 문제가 생길 수 있습니다.
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

# 프로젝트 설정(모델 이름 등)
import GameConfig as Config

# .env 로드 (OPENAI_API_KEY, OPENAI_MODEL 등)
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# 1) Tool 정의: LLM이 호출할 "함수"를 만든다.
# ---------------------------------------------------------------------------

# @tool 데코레이터를 붙이면,
# - 이 함수는 "LLM이 호출 가능한 도구"로 등록됩니다.
# - 함수 시그니처(a: int, b: int)와 docstring 설명이
#   LLM에게 전달되어 "어떤 인자로 어떻게 호출할지"를 결정할 수 있게 됩니다.
@tool
def add(a: int, b: int) -> int:
    """두 정수 a와 b를 더한 결과를 반환합니다."""

    return a + b


# ---------------------------------------------------------------------------
# 2) Graph State 정의: 대화 메시지를 State에 담는다.
# ---------------------------------------------------------------------------

# LangGraph에서 State는 "그래프 실행 중 공유되는 데이터"입니다.
# Tool Calling 튜토리얼에서 가장 중요한 State는 messages입니다.
#
# messages에는 아래 것들이 쌓입니다.
# - 사용자 메시지 (user)
# - LLM의 응답 메시지 (assistant)
# - LLM이 도구를 호출하겠다는 tool call 정보
# - 도구 실행 결과(tool result) 메시지
#
# add_messages는 "이 노드가 반환한 messages를 기존 messages에 누적"해주는 헬퍼입니다.
# 즉, 각 노드가 {"messages": [새 메시지]} 형태로 반환하면,
# LangGraph가 자동으로 messages 리스트 뒤에 append 해 줍니다.
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 3) LLM 준비 + tools 바인딩
# ---------------------------------------------------------------------------

# Tool Calling의 핵심은 "LLM이 tool을 호출할 수 있도록" LLM에 tools를 바인딩하는 것입니다.
#
# .bind_tools([add])를 하면:
# - LLM은 답변을 만들 때 "add 도구를 호출할지"를 선택할 수 있습니다.
# - 필요하다고 판단하면, 일반 텍스트 답변 대신 "tool 호출 계획"(tool_calls)을 출력합니다.
model_name = os.getenv("OPENAI_MODEL", getattr(Config, "llm_model_name", "gpt-4o-mini"))
llm = ChatOpenAI(model=model_name, temperature=0).bind_tools([add])


# ---------------------------------------------------------------------------
# 4) 노드 정의: assistant 노드 / tools 노드
# ---------------------------------------------------------------------------

# (A) assistant 노드
# - 현재 messages를 입력으로 받아 LLM을 한 번 호출하고,
# - LLM이 만든 메시지(AIMessage)를 messages에 추가합니다.
#
# LLM의 출력은 크게 두 가지 중 하나입니다.
# 1) 도구가 필요 없으면: 그냥 자연어 답변(content)을 가진 AIMessage
# 2) 도구가 필요하면: tool_calls가 들어있는 AIMessage
#
# 중요한 점:
# - 이 시점에서는 "도구를 실제로 실행"하지 않습니다.
# - 도구를 실행하는 건 다음 단계(= tools 노드)입니다.
def assistant(state: State) -> dict[str, Any]:
    ai_msg = llm.invoke(state["messages"])
    return {"messages": [ai_msg]}


# (B) tools 노드
# - ToolNode는 "LLM이 요청한 tool_calls를 읽고"
# - 실제 파이썬 함수를 실행한 뒤
# - 그 결과를 ToolMessage로 만들어 messages에 추가합니다.
#
# 즉, "도구 실행"은 여기서 일어납니다.
tool_node = ToolNode([add])


def route_after_assistant(state: State) -> str:
    """assistant 다음에 어디로 갈지 결정하는 '조건 함수'입니다.

    핵심 아이디어
    - 마지막 메시지에 tool_calls가 있으면: tools 노드로 가서 도구를 실행
    - tool_calls가 없으면: END로 종료

    참고
    - LangGraph는 tools_condition 같은 '기본 제공' 조건 함수도 있지만,
        이런 조건 함수는 우리가 직접 만들어서 원하는 규칙으로 바꿀 수 있습니다.
    """

    # 현재까지 누적된 메세지를 가져오고, 없으면 빈 리스트.
    messages = state.get("messages", [])

    # 메세지가 있으면 가장 마지막 메세지를 가져오고, 없으면 None.
    last = messages[-1] if messages else None

    # 마지막 메시지에 tool_calls가 있으면 가져오고, 없으면 None.
    tool_calls = getattr(last, "tool_calls", None)

    # tool_calls가 있으면 "tools"로, 없으면 "end"로 리턴.
    return "tools" if tool_calls else "end"


# ---------------------------------------------------------------------------
# 5) 그래프 구성: assistant ↔ tools 루프
# ---------------------------------------------------------------------------

# 그래프는 아래처럼 동작합니다.
#
# START
#   ↓
# assistant  --(tool_calls 있으면)--> tools --(결과를 messages에 추가)--> assistant
#    |\
#    | (tool_calls 없으면)
#    v
#   END
#
# 여기서 핵심이 tools_condition 입니다.
# - tools_condition은 "assistant 노드가 방금 만든 마지막 메시지"를 보고
#   tool_calls가 있으면 "tools"로 라우팅
#   없으면 END로 라우팅합니다.
# - 즉, "툴을 쓸지 말지"를 LLM이 결정하면,
#   LangGraph가 그 결정을 따라 자동으로 다음 노드를 선택해줍니다.

builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")

# END를 "명시적으로" 연결하고 싶으면, 아래처럼 path_map을 적어주면 됩니다.
# - return 값("tools"/"end")을 어떤 노드로 연결할지 눈으로 확인 가능
builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "tools": "tools",
        "end": END,
    },
)
builder.add_edge("tools", "assistant")

graph = builder.compile()


# ---------------------------------------------------------------------------
# 6) 실행: 질문을 던지고 마지막 답변을 출력
# ---------------------------------------------------------------------------


def _print_debug(messages: list[Any]) -> None:
    """학습용: messages에 어떤 이벤트가 쌓였는지 간단히 출력합니다.

    초보자 입장에서는 Tool Calling이 "마법"처럼 느껴질 수 있어서,
    실제로는 messages 안에 어떤 항목들이 순서대로 들어오는지 보여주는 것이 이해에 도움이 됩니다.

    - AIMessage에 tool_calls가 생기면 "도구를 호출하려고 했구나"를 알 수 있고
    - ToolMessage가 생기면 "도구 실행 결과가 들어왔구나"를 알 수 있습니다.
    """

    print("\n[디버그] messages 흐름(요약)")
    for i, m in enumerate(messages, start=1):
        t = type(m).__name__
        content = getattr(m, "content", None)
        tool_calls = getattr(m, "tool_calls", None)
        print(f"- #{i} {t}")
        if tool_calls:
            print(f"    tool_calls={tool_calls}")
        if isinstance(content, str) and content.strip():
            print(f"    content={content[:160]}")


def main() -> None:
    print("LangGraph Tool Calling 튜토리얼")
    print("- 예: '12와 34 더해줘' / 'a=7, b=9 더한 값 알려줘'\n")

    while True:
        question = input("질문(Enter로 종료): ").strip()
        if not question:
            break

        # graph.invoke에 State를 넣어 실행합니다.
        # messages는 ("user", "텍스트") 같은 튜플로 넣어도 되고,
        # HumanMessage 같은 객체로 넣어도 됩니다.
        result = graph.invoke({"messages": [("user", question)]})

        messages = result.get("messages", [])
        last = messages[-1] if messages else None

        # 학습용 디버그 출력(싫으면 주석 처리)
        _print_debug(messages)

        if isinstance(last, AIMessage):
            print(f"\n[최종 답변]\n{last.content}")
        else:
            print(f"\n[최종 답변]\n{last}")


if __name__ == "__main__":
    main()
