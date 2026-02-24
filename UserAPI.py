from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json
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

    #return {"message": f"{uid}님이 로그인 했습니다." }
    return {"message": "안녕하세요. 저는 쿠팡 유출 사태와 관련된 질문을 처리할 수 있습니다. 무엇이 궁금하신가요?"}

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

    multi_queries = res.get("multi_queries") or []
    meta = {
        "multi_queries": multi_queries,
        "multi_query_count": len(multi_queries),
        "ragas": res.get("ragas"),
    }

    return {
        "message": ai_msg,
        "meta": meta,
    }

@traceable
@router.get("/userchat_async")
async def userchat_async(uid: str, message: str):
    """
    채팅 티키타카 (스트리밍 버전)
    - 즉시 응답(StreamingResponse)을 반환

    Args:
        uid: 사용자 ID
        message: 사용자 메시지
        emit_node: True면 노드 진행 이벤트를 함께 스트리밍(NDJSON)

    Returns:
        - emit_node=False(기본): 최종 답변 토큰을 raw text로 스트리밍
        - emit_node=True: NDJSON 스트림
            {"type":"node","node":"multi_query","status":"start"}\n
            {"type":"token","text":"..."}\n
    """

    print( f"[{uid}의 질문(Stream)]: {message}" )

    # 1. 유저 및 에이전트 준비
    user = User.DB.get_or_make_user( uid )
    agent = Agent( uid )

    # 2. 비동기 제너레이터 함수 정의 (여기서 로직과 출력을 동시에 처리)
    async def response_generator():
        full_answer = ""
        current_node = None
        qa_result = {
            "query_count" : 0,
            "ragas" : "평가 점수가 없습니다."
        }

        def _ndjson(obj: dict) -> str:
            return json.dumps(obj, ensure_ascii=False) + "\n"

        # LangGraph 이벤트 스트림을 받아서 final_answer 노드의 토큰만 필터링
        events = await agent.run_qa_astream_events(message)

        async for ev in events:

            # event or type: 무슨 일이 발생 했는지. (on_chain_start, on_chain_end 등등.....)
            ev_type = ev.get("event") or ev.get("type")

            # metadata: "이 이벤트가 어떤 컨텍스트에서 발생했는지"를 담는 부가 정보
            # - 이벤트 종류에 따라 metadata가 없을 수도 있으므로(None), 항상 dict로 맞춰서 처리
            # - 여기서는 아래에서 그래프 노드 이름을 뽑기 위해 사용
            #   (metadata['langgraph_node'] 또는 metadata['node'])
            metadata = ev.get("metadata") or {}
            
            # 실제로 실행된 runnable/컴포넌트 이름.
            # 이번 이벤트를 발생시킨 실제 runnable/컴포넌트 이름.
            # RunnableSequence, PromptTemplate, ChatOpenAI, EnsembleRetriever처럼 노드 내부 구성요소 이름이 들어올 때도 있음
            ev_name = ev.get("name")

            # 이 이벤트가 속한 "그래프 노드 이름"
            # - node_name: question/multi_query/tool_call/final_answer/summary 같은 LangGraph 노드 이름
            # - ev_name:   노드 자체일 수도 있고, 노드 내부 구성요소(PromptTemplate/ChatOpenAI 등)일 수도 있음
            #
            # 목적: "노드가 바뀌는 순간"(A -> B)만 Unity로 알리고 싶음
            # 그래서 아래처럼 '노드 자체 시작 이벤트'만 골라냅니다.
            node_name = metadata.get("langgraph_node") or metadata.get("node")            

            # 이벤트 payload는 ev['data'] 아래에 들어오는 경우가 많음
            # - stream 이벤트에서는 보통 data['chunk']에 "토큰 조각"(str 또는 MessageChunk 유사 객체)이 담김
            data = ev.get("data") or {}
            chunk = data.get("chunk")

            # 1) 노드 전환 이벤트를 뽑는 기준
            # - on_chain_start: 어떤 체인(=노드/내부 runnable)이 시작될 때
            # - node_name이 있어야 "그래프의 어떤 노드인지" 알 수 있음
            # - ev_name == node_name 인 경우만 "그래프 노드 자체"의 시작으로 간주
            #   (예: node=multi_query, name=RunnableSequence 는 노드 내부라서 제외)
            is_chain_start = ev_type == "on_chain_start"
            is_chain_end = ev_type == "on_chain_end"
            has_node_name = bool(node_name)
            is_graph_node_start = has_node_name and (ev_name == node_name)

            if is_chain_end and node_name == "graph_end":
                # 마지막 노드에 도달하면 실행 정보를 보여줄 정보를 수집함
                if node_name == "graph_end":
                    output = data.get( "output" )
                    if isinstance( output, dict ):
                        mq = output.get( "multi_queries", [] )
                        qa_result[ "query_count" ] = len(mq)
                        qa_result[ "ragas" ] = output.get( "ragas", "평가 점수가 없습니다" )

            elif is_chain_start and is_graph_node_start:

                # 2) 같은 노드 이벤트는 중복 전송하지 않음
                if node_name != current_node:
                    current_node = node_name
                    # 노드에 커스텀 메타데이터를 붙여둔 경우(예: add_node(..., metadata={...}))
                    # Unity로 함께 전달할 값을 여기서 꺼내서 포함시킬 수 있음
                    payload = {"type": "node", "node": node_name, "status": "start"}

                    unity_label = metadata.get("unity_label")
                    if unity_label is not None:
                        payload["meta"] = unity_label
                        print( unity_label )

                    yield _ndjson(payload)

            # final_answer 노드에서 생성되는 토큰만 유니티로 스트리밍
            if node_name and node_name != "final_answer":
                continue
            if ev_type not in ("on_chat_model_stream", "on_llm_stream"):
                continue

            # chunk가 없으면(다른 타입의 이벤트이거나 비어있으면) 토큰으로 보낼 게 없으니 스킵
            if chunk is None:
                continue

            # chunk가 str이면 그대로 사용, 객체면 .content를 꺼내서 텍스트로 변환
            text = chunk if isinstance(chunk, str) else (getattr(chunk, "content", "") or "")

            # 빈 문자열이면 전송할 의미가 없어서 스킵
            if not text:
                continue

            full_answer += text
            await asyncio.sleep(0.01)

            yield _ndjson({"type": "token", "text": text})

        # 스트림이 끝난 후 최종 답변과 실행 결과 정보를 넘김 (멀티 쿼리 수, RAGAS 점수...)
        # metal에 넣지 말고 eval로 따로 넣자.
        yield _ndjson({"type": "qa_result", "eval": qa_result})
        await asyncio.sleep(0.05)  # 마지막 청크가 전송될 때까지 대기
        yield _ndjson({"type":"end"})

    # 3. StreamingResponse로 반환
    # - 기본: raw text 스트림
    
    #media_type = "text/event-stream" # 단순 텍스트 청크 스트림
    media_type = "application/x-ndjson"
    return StreamingResponse(response_generator(), media_type=media_type)

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
    #return login( uid )
    
    return "안녕하세요. 저는 쿠팡 유출 사태와 관련된 질문을 처리할 수 있습니다. 무엇이 궁금하신가요?"



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
