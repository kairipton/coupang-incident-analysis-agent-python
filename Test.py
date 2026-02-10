import os
import dotenv
dotenv.load_dotenv()

import GameAPI
import asyncio
from fastapi.responses import StreamingResponse
import Utils.Utils as Utils


# [초기화] 서버 시작 전 리소스 로드 (DB 등)
#print("=== 시스템 초기화 중... ===")
#vector_db = Utils.load_vector_db()
#main.on_startup()
#print("=== 초기화 완료 ===")

# ----------------------------------------------------------------
# 1. [동기 모드] 순수 Python 함수 호출 (Blocking)
# - GameAPI.userchat()을 사용
# - asyncio 전혀 사용 안 함
# ----------------------------------------------------------------
def run_sync_test():
    print("\n[TEST MODE] 1. 동기(Sync) 테스트 시작")
    uid = "test_user_sync"

    # 1. 로그인
    welcome_pack = GameAPI.login(uid)
    print(f"\n[SYSTEM]: {welcome_pack['message']}")

    while True:
        user_input = input("\n[나(Sync)]: ")
        if user_input.lower() in ["exit", "quit"]:
            GameAPI.logout(uid)
            break

        if user_input.lower() in ["reset"]:
            GameAPI.reset(uid)
            continue
        
        # 2. 동기 함수 호출 (그냥 함수 부르듯이)
        # main.userchat은 dict를 리턴합니다.
        response = GameAPI.userchat(uid, user_input)
        
        # 3. 결과 출력
        # 동기 모드는 스트리밍이 아니므로 완성된 문장이 한 번에 옵니다.
        ai_msg = response["message"]
        #o2 = response["o2"]
        #phase = response["phase"]
        
        print(f"[AI]: {ai_msg}")
        #print(f"(Debug: 산소={o2}, 페이즈={phase})")

# ----------------------------------------------------------------
# 2. [비동기 모드] 스트리밍 호출 (Non-Blocking)
# - main.userchat_async()를 사용
# - async/await 사용
# ----------------------------------------------------------------
async def run_async_test():
    print("\n[TEST MODE] 2. 비동기(Async) 스트리밍 테스트 시작")
    uid = "test_user_async"

    # 1. 로그인 (로그인은 동기 함수지만 여기서 불러도 됨)
    welcome_pack = GameAPI.login(uid)
    print(f"\n[SYSTEM]: {welcome_pack['message']}")

    while True:
        # input()은 동기 함수라 흐름을 막지만 테스트니까 사용
        user_input = input("\n[나(Async)]: ")
        if user_input.lower() in ["exit", "quit"]:
            GameAPI.logout(uid)
            break

        if user_input.lower() in ["reset"]:
            GameAPI.reset(uid)
            continue

        print("[AI]: ", end="", flush=True)

        # 2. 비동기 함수 호출 (await 필수)
        # GameAPI.userchat_async는 StreamingResponse 객체를 리턴
        response: StreamingResponse = await GameAPI.userchat_async(uid, user_input)

        # 3. 스트림 데이터 소비
        # body_iterator를 비동기 루프로 읽어옴
        async for chunk in response.body_iterator:
            # chunk가 bytes일 수도, str일 수도 있음 (FastAPI 구현에 따라 다름)
            # 현재 코드상 yield chunk (str) 하므로 str로 올 것임
            text_chunk = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
            print(text_chunk, end="", flush=True)
        
        print() # 줄바꿈

if __name__ == "__main__":
    run_sync_test()
    #asyncio.run(run_async_test())