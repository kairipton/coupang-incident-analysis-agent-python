"""MultiAgent.py

멀티 에이전트(Multi-Agent) 패턴을 LangGraph로 구현하는 튜토리얼입니다.

핵심 목표
- 역할이 다른 두 에이전트(분석, 답변)를 각각 독립적인 StateGraph로 만든다.
- 오케스트레이터(조율자) 그래프가 두 에이전트를 "노드"로 등록해서 순서대로 호출한다.
- "에이전트 = 또 하나의 그래프"라는 개념을 체득한다.

멀티 에이전트의 핵심 아이디어
- 단일 에이전트: LLM 하나가 검색·판단·답변을 모두 처리
- 멀티 에이전트: 각 에이전트가 한 가지 역할만 담당하고, 오케스트레이터가 흐름을 조율
  └ 역할 분리 → 각 에이전트를 독립적으로 교체/개선 가능

이 튜토리얼의 흐름
    [오케스트레이터 그래프]
            |
       ┌────▼────┐
       │분석 에이전트│  ← 질문을 받아 핵심 사실/키워드 추출
       └────┬────┘
            |
       ┌────▼────┐
       │답변 에이전트│  ← 분석 결과를 받아 최종 답변 생성
       └─────────┘

준비물
- 프로젝트 루트의 .env에 OPENAI_API_KEY 필요
- (선택) OPENAI_MODEL 설정 (없으면 GameConfig.llm_model_name 사용)

실행 (프로젝트 루트에서)
- ./.venv/Scripts/python.exe ./Tutorial/MultiAgent.py

이 튜토리얼에서 배우는 것
1) 에이전트 간 공유되는 State를 어떻게 설계하는지
2) 각 에이전트를 독립 StateGraph로 만드는 방법
3) 서브그래프를 오케스트레이터의 "노드"로 등록하는 핵심 패턴
4) State의 특정 필드만 서브그래프에 전달/수신하는 방법(입출력 매핑)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# 0) 프로젝트 루트 고정 + 환경 설정
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

import GameConfig as Config

load_dotenv(PROJECT_ROOT / ".env")

model_name = os.getenv("OPENAI_MODEL", getattr(Config, "llm_model_name", "gpt-4o-mini"))
llm = ChatOpenAI(model=model_name, temperature=0)


# ---------------------------------------------------------------------------
# 1) 공유 State 정의
#
# 멀티 에이전트에서 State는 "에이전트 간의 공용 메모장"입니다.
# - question       : 사용자가 입력한 원본 질문
# - analysis_result: 분석 에이전트가 추출한 핵심 정보
# - final_answer   : 답변 에이전트가 생성한 최종 답변
#
# 각 에이전트(서브그래프)는 State 전체를 받지만,
# 자신의 역할에 해당하는 필드만 읽고, 자신이 담당하는 필드만 채워서 반환합니다.
# ---------------------------------------------------------------------------

class State(TypedDict):
    question: str         # 오케스트레이터가 채움
    analysis_result: str  # 분석 에이전트가 채움
    final_answer: str     # 답변 에이전트가 채움


# ---------------------------------------------------------------------------
# 2) 분석 에이전트 (Analyst Agent)
#
# 역할: 사용자 질문을 받아서 "핵심 사실/키워드"만 간결하게 추출한다.
# 입력: state["question"]
# 출력: state["analysis_result"]
#
# 독립적인 StateGraph이지만, 노드가 하나(analyze)인 단순한 구조입니다.
# 실제 프로젝트에서는 이 안에 RAG 검색, Tool Calling 등을 추가할 수 있습니다.
# ---------------------------------------------------------------------------

def analyst_node(state: State) -> dict:
    """질문에서 핵심 사실/키워드를 추출하는 노드."""

    print("\n[분석 에이전트] 질문 분석 중...")

    messages = [
        SystemMessage(content=(
            "당신은 질문 분석 전문가입니다.\n"
            "사용자의 질문을 읽고, 답변에 필요한 핵심 사실과 키워드를 3~5개 항목으로 정리해주세요.\n"
            "형식: '- 항목1\\n- 항목2\\n...' (다른 말은 붙이지 마세요)"
        )),
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)

    print(f"[분석 에이전트] 추출 결과:\n{response.content}\n")

    # 자신이 담당하는 필드(analysis_result)만 반환합니다.
    return {"analysis_result": response.content}


# 분석 에이전트를 독립적인 StateGraph로 구성합니다.
analyst_builder = StateGraph(State)
analyst_builder.add_node("analyze", analyst_node)
analyst_builder.add_edge(START, "analyze")
analyst_builder.add_edge("analyze", END)

# .compile()로 실행 가능한 그래프 객체를 만듭니다.
# 이 객체가 오케스트레이터의 "노드"로 등록됩니다.
analyst_graph = analyst_builder.compile()


# ---------------------------------------------------------------------------
# 3) 답변 에이전트 (Writer Agent)
#
# 역할: 분석 결과를 토대로 사용자에게 전달할 최종 답변을 생성한다.
# 입력: state["question"] + state["analysis_result"]
# 출력: state["final_answer"]
#
# 분석 에이전트의 결과(analysis_result)를 프롬프트에 포함시켜서
# 더 정확하고 구조화된 답변을 만들어냅니다.
# ---------------------------------------------------------------------------

def writer_node(state: State) -> dict:
    """분석 결과를 바탕으로 최종 답변을 생성하는 노드."""

    print("[답변 에이전트] 최종 답변 생성 중...")

    messages = [
        SystemMessage(content=(
            "당신은 친절한 답변 작성 전문가입니다.\n"
            "아래 '분석 결과'를 참고해서, 사용자의 질문에 대한 명확하고 친절한 답변을 작성해주세요.\n"
            "분석 결과를 그대로 나열하지 말고, 자연스러운 문장으로 풀어서 설명해주세요."
        )),
        HumanMessage(content=(
            f"[사용자 질문]\n{state['question']}\n\n"
            f"[분석 결과]\n{state['analysis_result']}"
        )),
    ]

    response = llm.invoke(messages)

    print(f"[답변 에이전트] 생성 완료.\n")

    return {"final_answer": response.content}


# 답변 에이전트도 독립적인 StateGraph로 구성합니다.
writer_builder = StateGraph(State)
writer_builder.add_node("write", writer_node)
writer_builder.add_edge(START, "write")
writer_builder.add_edge("write", END)

writer_graph = writer_builder.compile()


# ---------------------------------------------------------------------------
# 4) 오케스트레이터 (Orchestrator)
#
# 핵심 패턴:
#     orchestrator.add_node("analyst", analyst_graph.compile())
#
# - 서브그래프를 .compile()한 객체를 그냥 노드로 등록합니다.
# - 오케스트레이터 입장에서는 내부 구조를 알 필요 없이 "노드 하나"일 뿐입니다.
# - State가 공유되므로, 분석 에이전트가 채운 analysis_result를
#   답변 에이전트가 그대로 읽을 수 있습니다.
#
# 흐름:
#   START → analyst(분석 에이전트) → writer(답변 에이전트) → END
# ---------------------------------------------------------------------------

orchestrator_builder = StateGraph(State)

# 서브그래프를 노드로 등록 — 멀티 에이전트의 핵심 패턴입니다.
orchestrator_builder.add_node("analyst", analyst_graph)
orchestrator_builder.add_node("writer", writer_graph)

orchestrator_builder.add_edge(START, "analyst")
orchestrator_builder.add_edge("analyst", "writer")
orchestrator_builder.add_edge("writer", END)

orchestrator = orchestrator_builder.compile()

# 그래프 구조를 ASCII로 출력
print(orchestrator.get_graph(xray=True).draw_ascii())


# ---------------------------------------------------------------------------
# 5) 실행
# ---------------------------------------------------------------------------

def main() -> None:
    print("멀티 에이전트 튜토리얼")
    print("분석 에이전트 → 답변 에이전트 순으로 처리됩니다.\n")

    while True:
        question = input("질문 (Enter로 종료): ").strip()
        if not question:
            break

        # 초기 State: question만 채워서 시작합니다.
        # analysis_result, final_answer는 각 에이전트가 순서대로 채워줍니다.
        initial_state: State = {
            "question": question,
            "analysis_result": "",
            "final_answer": "",
        }

        result = orchestrator.invoke(initial_state)

        print("=" * 60)
        print("[최종 답변]")
        print(result["final_answer"])
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
