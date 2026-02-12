"""ToolCallingTutorial.py

목표
- LangChain의 Tool Calling(에이전트가 함수를 선택/호출)을 가장 단순한 형태로 익힌다.
- 같은 Tool Calling을 LCEL(Runnable 파이프라인)과 결합해 "동적 컨텍스트"를 주입하는 방법을 익힌다.

실행 준비
1) 프로젝트 루트의 .env 파일에 키가 있어야 합니다.
   - OPENAI_API_KEY=... (필수)
   - (선택) OPENAI_MODEL=gpt-4o-mini  또는 원하는 모델명
   - (선택) LANGSMITH_API_KEY=... 등 LangSmith 관련 변수

2) 실행
     - Windows PowerShell:
         ./.venv/Scripts/python.exe ./Tutorial/ToolCallingTutorial.py

주의
- 이 파일은 "튜토리얼" 목적이라, 로직을 짧고 명확하게 유지했습니다.
- Tool Calling은 "프롬프트 문자열"만으로 되는 게 아니라, create_agent에 tools를 등록하고
  agent.invoke(...)에 상태(state)를 맞춰 넣는 방식으로 동작합니다.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ---------------------------------------------------------------------------
# 공통: 프로젝트 루트 기준으로 동작하게 고정
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 공통: 환경변수 로드 + LLM 생성
# ---------------------------------------------------------------------------

def build_llm() -> ChatOpenAI:
    """프로젝트 루트의 .env를 읽어 ChatOpenAI 모델을 생성합니다."""

    load_dotenv(PROJECT_ROOT / ".env")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # OPENAI_API_KEY는 langchain_openai가 내부에서 읽습니다.
    return ChatOpenAI(model=model)


# ---------------------------------------------------------------------------
# 1) Tool Calling 기본 튜토리얼
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""

    return a + b


@tool
def now_iso() -> str:
    """현재 시간을 ISO 문자열로 반환합니다."""

    return datetime.now().isoformat(timespec="seconds")


@tool
def system_info() -> str:
    """OS/파이썬 간단 정보를 반환합니다."""

    return f"os={platform.system()} {platform.release()}, python={platform.python_version()}"


def extract_last_ai_text(result: Any) -> str:
    """create_agent 결과(state dict)에서 마지막 AI 메시지 텍스트만 뽑습니다."""

    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content

    return str(messages[-1]) if messages else str(result)


def tutorial_basic_tool_calling() -> None:
    """(튜토리얼 1) Tool Calling만으로 에이전트를 만드는 가장 기본 형태."""

    llm = build_llm()
    tools = [add, now_iso, system_info]

    system_prompt = (
        "당신은 시스템 관리자를 돕는 AI 에이전트입니다.\n"
        "필요하면 tools를 사용해서 정확히 답하세요.\n"
        "계산은 add, 시간은 now_iso, 시스템 정보는 system_info를 사용하세요."
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    print("\n[튜토리얼 1] Tool Calling 기본")
    print("- 질문 예: '지금 시간 알려줘' / '2 더하기 40은?' / '내 시스템 정보 알려줘'")

    while True:
        question = input("\n질문(Enter로 종료): ").strip()
        if not question:
            break

        result = agent.invoke({"messages": [("user", question)]})
        answer = extract_last_ai_text(result)
        print(f"\n[답변]\n{answer}")


# ---------------------------------------------------------------------------
# 2) LCEL + Tool Calling 발전형 튜토리얼
# ---------------------------------------------------------------------------


KB: Dict[str, str] = {
    "cache": "캐시는 문제 원인 파악 후 비우는 게 안전합니다. 서비스 영향 범위를 확인하세요.",
    "logs": "로그 회전 전에는 디스크 사용량, 보관 정책, 권한을 확인하세요.",
    "ip": "IP 차단은 오탐 위험이 있어, 적용 전/후 모니터링과 롤백 절차를 준비하세요.",
}


def keyword_retrieve(question: str) -> str:
    """아주 단순한 '검색기' 예시(튜토리얼용)."""

    q = question.lower()
    hits = []
    for k, v in KB.items():
        if k in q:
            hits.append(f"- {k}: {v}")

    return "\n".join(hits) if hits else "(관련 컨텍스트 없음)"


def tutorial_advanced_lcel_plus_tools() -> None:
    """(튜토리얼 2) LCEL로 컨텍스트 생성/주입 + Tool Calling."""

    llm = build_llm()
    tools = [add, now_iso, system_info]

    base_system_prompt = (
        "당신은 시스템 관리자를 돕는 AI 에이전트입니다.\n"
        "아래 [Context]가 있으면 참고하고, 필요하면 tools를 호출해서 답하세요.\n"
        "답변은 짧고 실행 가능한 형태로 쓰세요."
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=base_system_prompt,
    )

    def build_state(x: Dict[str, str]) -> Dict[str, Any]:
        system_msg = f"[Context]\n{x['context']}"
        return {
            "messages": [
                ("system", system_msg),
                ("user", x["question"]),
            ]
        }

    pipeline = (
        RunnablePassthrough.assign(context=(lambda x: keyword_retrieve(x["question"])))
        | RunnableLambda(build_state)
        | agent
        | RunnableLambda(extract_last_ai_text)
    )

    print("\n[튜토리얼 2] LCEL + Tool Calling 발전형")
    print("- 키워드(cache/logs/ip)가 들어가면 Context가 자동 주입됩니다.")
    print("- 예: '로그 정리 어떻게 해?' / 'cache 비워도 돼?' / 'IP 차단 방법?' / '2+3 계산'\n")

    while True:
        question = input("\n질문(Enter로 종료): ").strip()
        if not question:
            break

        answer_text = pipeline.invoke({"question": question})
        print(f"\n[답변]\n{answer_text}")


def main() -> None:
    print("ToolCallingTutorial 실행")
    print("1) Tool Calling 기본 튜토리얼")
    print("2) LCEL + Tool Calling 발전형 튜토리얼")

    choice = input("\n선택 (1/2): ").strip()
    if choice == "2":
        tutorial_advanced_lcel_plus_tools()
    else:
        tutorial_basic_tool_calling()


if __name__ == "__main__":
    main()
