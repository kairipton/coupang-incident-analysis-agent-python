from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
from langsmith import traceable
from langchain_core.messages import BaseMessage, AIMessage

from Utils import User
from AI.Agent import Agent
import Utils.Utils as Utils
from pprint import pprint
from typing import Any

router = APIRouter()

# 전역 리소스 (startup 이벤트에서 초기화)
# - FastAPI 라우트 함수들에서 참조하므로, 모듈 스코프에 "변수"는 먼저 만들어 두고
#   실제 값(벡터 DB)은 서버 시작 시점(startup)에 채웁니다.
# - 타입 힌트/린터 관점에서도 NameError를 예방할 수 있습니다.
vector_db = Utils.load_vector_db()

@router.get("/ping")
def ping():
    return {"message": "퐁~!"}

@router.get( "/login" )
def login(uid:str):
    """
    사용자 로그인

    Args:
        uid (str): 사용자 ID

    Returns:
        dict: 첫 페이즈의 환영 메세지
    """
    user = User.DB.get_or_make_user( uid )

    print( f"{uid}님이 로그인했습니다." )
    
    #return {"message": f"{user.uid}님 환영합니다."}

    #welcome_msg = GameManager.get_welcome_msg(user)
    #user.history.add_ai_message( welcome_msg )

    return {"message": f"{uid}님이 로그인 했습니다." }

@router.get("/logout")
def logout(uid:str):
    """
    사용자 로그아웃

    Args:
        uid (str): 사용자 ID
    """
    User.DB.delete_user( uid )

    print( f"{uid}님이 로그아웃했습니다." )

    return {"message": f"{uid}님 로그아웃되었습니다."}

@traceable
@router.get("/userchat")
def userchat(uid:str, message: str):
    """
    채팅 티키타카
    실제 플레이 로직이 진행됨

    Args:
        uid (str): 사용자 ID
        message (str): 사용자 메시지

    Returns:
        dict: AI의 답변(message), 산소량(o2), 현재 페이즈(phase)
    """

    print( f"[{uid}의 질문]: {message}" )

    user = User.DB.get_or_make_user( uid )
    agent = Agent( uid )

    res = agent.run_qa( message )
    ai_msg=  __get_ai_message( res )
    
    return {
        "message": ai_msg, 
    }

@traceable
@router.get("/userchat_async")
async def userchat_async(uid:str, message: str):
    """
    채팅 티키타카 (스트리밍 버전)
    - 즉시 응답(StreamingResponse)을 반환
    """

    print( f"[{uid}의 질문(Stream)]: {message}" )

    # 1. 유저 및 에이전트 준비
    user = User.DB.get_or_make_user( uid )
    agent = Agent( uid )

    # 2. 비동기 제너레이터 함수 정의 (여기서 로직과 출력을 동시에 처리)
    async def response_generator():
        full_answer = ""

        # LangGraph 이벤트 스트림을 받아서 final_answer 노드의 토큰만 필터링
        events = await agent.run_qa_astream_events(message)

        async for ev in events:
            metadata = ev.get("metadata") or {}
            node_name = metadata.get("langgraph_node") or metadata.get("node")

            # final_answer 노드에서 생성되는 토큰만 유니티로 스트리밍
            if node_name and node_name != "final_answer":
                continue

            ev_type = ev.get("event") or ev.get("type")
            if ev_type not in ("on_chat_model_stream", "on_llm_stream"):
                continue

            data = ev.get("data") or {}
            chunk = data.get("chunk")

            if chunk is None:
                continue

            text = chunk if isinstance(chunk, str) else (getattr(chunk, "content", "") or "")
            if not text:
                continue

            full_answer += text
            await asyncio.sleep(0.01)
            yield text

    # 3. StreamingResponse로 반환 (Content-Type: text/event-stream)
    return StreamingResponse(response_generator(), media_type="text/event-stream")

@router.get("/reset")
def reset(uid:str):
    """
    현재 게임 상태를 리셋하고 처음부터 다시 시작한다.

    Args:
        uid (str): 사용자 ID

    Returns:
        dict: 첫 페이즈의 환영 메세지
    """

    logout( uid )
    return login( uid )



def __get_ai_message(response:dict[str, Any]) -> str:
    """
    LLM 응답중 AI의 마지막 메세지만 쏙 빼옴.

        Args:
            response (dict): LLM 응답 (랭그래프의 State)

        Returns:
            str: AI의 마지막 메시지 내용
    """
    # print( "------------ AI 응답 출력 시작 ------------")
    # pprint( response, indent=2, width=100 )
    # print( "------------ 키 목록 ------------")
    # print( response.keys() )
    # print( "------------ AI 메시지만 빼오기 ------------")

    msg = ""
    for key in response:
        if key == "messages":    
            messages: list[BaseMessage] = response.get( key, [] )
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    msg = msg.content
                    break

    return msg
