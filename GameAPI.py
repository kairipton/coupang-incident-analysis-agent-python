from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
from langsmith import traceable

#from GameManager import GameManager
from Utils import User
from AI.Agent import Agent
#import AI.Agent as Agent
import Utils.Utils as Utils


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

    return {"message": f"{uid}님이 로그인 했습니다.", "o2": user.o2, "phase": user.phase}

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
    agent = Agent( vector_db.as_retriever( k=6 ), uid, User.DB.get_history )

    res = agent.run_qa( message )
    ai_message = str(res)
    print( f"AI의 답변 -> {res}" )

    return {
         "message": f"{ai_message}", 
    }

@traceable
@router.get("/userchat_async")
async def userchat_async(uid:str, message: str):
    """
    채팅 티키타카 (스트리밍 버전)
    - 즉시 응답(StreamingResponse)을 반환
    """

    # [디버깅용] 유니티가 접속할 때 키가 있는지 콘솔에 찍어봅니다.
    #print(f"DEBUG: API KEY Status -> {os.getenv('LANGSMITH_API_KEY')}****") # <--- 추가

    print( f"[{uid}의 질문(Stream)]: {message}" )

    # 1. 유저 및 에이전트 준비
    user = User.DB.get_or_make_user( uid )
    agent = Agent( vector_db.as_retriever( k=6 ), uid, User.DB.get_history )

    # 2. 비동기 제너레이터 함수 정의 (여기서 로직과 출력을 동시에 처리)
    async def response_generator():
        full_answer = ""
        
        # [중요] run_type=2 (astream)을 사용하여 비동기 스트림 객체를 받음
        # AIAgent.py의 run_qa가 async/await를 지원하도록 되어 있어야 함
        ai_stream = await agent.run_qa_async( message, { "o2": user.o2, "phase" : user.phase } )
        
        # AI 답변 실시간 전송 루프
        is_success: bool = False
        async for chunk in ai_stream:
            full_answer += chunk
            await asyncio.sleep( 0.01 ) # 속도 조절
            yield chunk 

            # 반복분을 돌며 성공 여부 계속 체크
            # 성공 했다면 즉시 탈출.
            # 출력 되어야 하는 답변 메세지는 페이즈 시작 메세지로 대체 되므로 추가 전송 필요 없음
            if "||SUCCESS||" in full_answer:
                is_success = True
                break

            # await asyncio.sleep(0.01) # (선택) 너무 빠르면 여기서 속도 조절 가능

        # 태그를 제거한 순수 텍스트만 추출
        clean_msg = full_answer.replace("||SUCCESS||", "").replace("||FAIL||", "")

        # 페이즈가 넘어갈 경우 페이즈 시작 메세지가 리턴 됨
        #welcome_msg = GameManager.phase_process( user, is_success )
        print( f"현재 산소량: {user.o2}, 현재 페이즈: {user.phase}" )

        # 산소량, 페이즈 정보를 포함.
        #welcome_msg = f"||O:{user.o2}||" + f" ||P:{user.phase}||" + welcome_msg

        # 페이즈 시작 메세지가 있을 경우 ai 대화 내역에 해당 메세지를 넣고, 없으면 ai 메세지 그대로 사용
        #if welcome_msg != "":
        #    user.history.add_ai_message( welcome_msg )

        #    msg = f"\n\n{welcome_msg}"
        #    for c in msg:
        #        yield c
        #else:
        user.history.add_ai_message( clean_msg )

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