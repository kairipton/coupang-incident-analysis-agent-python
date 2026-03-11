import os
import logging
import dotenv

# [핵심] 실행 위치가 어디든 상관없이, main.py 파일 옆에 있는 .env를 강제로 찾아냅니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# .env 파일 로드
is_loaded = dotenv.load_dotenv(ENV_PATH)

from langsmith import traceable

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio

from Utils import User
import AI.Agent as Agent
import Utils.Utils as Utils
import router as API

app = FastAPI()

app.include_router( API.router )

# 전역 리소스 (startup 이벤트에서 초기화)
# - FastAPI 라우트 함수들에서 참조하므로, 모듈 스코프에 "변수"는 먼저 만들어 두고
#   실제 값(벡터 DB)은 서버 시작 시점(startup)에 채웁니다.
# - 타입 힌트/린터 관점에서도 NameError를 예방할 수 있습니다.
vector_db = None

app.add_middleware(
    CORSMiddleware,
    # CORS(Cross-Origin Resource Sharing) 설정
    # - Unity에서 FastAPI로 HTTP 요청을 보낼 때, 실행 환경에 따라 "다른 출처(Origin)"로 인식될 수 있음
    #   예) Unity WebGL(브라우저에서 실행)은 페이지의 Origin(도메인/포트/프로토콜)에서 API(서버)로 요청을 보냄
    #       이때 브라우저 보안 정책(Same-Origin Policy) 때문에 서버가 CORS 허용 응답 헤더를 주지 않으면
    #       네트워크 요청이 "브라우저에서 차단"되어 Unity 쪽 코드가 응답을 받지 못함
    # - 또한 Unity에서 JSON 전송(예: Content-Type: application/json) 같은 요청은 종종 프리플라이트(OPTIONS) 요청이 먼저 발생함
    #   프리플라이트까지 통과하려면 서버가 OPTIONS를 포함한 메서드/헤더를 허용한다고 명시해야 함
    #
    # 개발 단계에선 편의상 전체 허용("*")을 쓰지만, 운영 환경에선 실제 Unity가 호스팅되는 Origin만 허용하는 것이 안전함.
    allow_origins=["*"],     # 어떤 Origin에서 오든 허용 (개발용으로 편함; 운영에선 특정 도메인으로 제한 권장)
    allow_methods=["*"],     # 프리플라이트 포함: GET/POST/OPTIONS 등 모든 메서드 허용
    allow_headers=["*"],     # 프리플라이트 포함: Content-Type/Authorization 등 모든 헤더 허용
)


@app.on_event("startup")
def on_startup():
        """서버 시작 시 1회 실행되는 초기화 훅(startup event).

        초보자용 설명
        - `@app.on_event("startup")`는 "이 함수는 서버가 시작할 때 자동으로 실행해줘"라고
            FastAPI(app)에 *등록*하는 문법입니다. (파이썬의 데코레이터)
        - 여기서 말하는 `startup`은 uvicorn 전용 이벤트가 아니라,
            ASGI 표준의 "lifespan"(앱 생명주기)에서 정의된 시작/종료 시점 이벤트입니다.

        누가/언제 호출하나요?
        - uvicorn 같은 ASGI 서버가 앱(FastAPI)을 실행할 때,
            "이제 앱 시작한다"라는 신호(lifespan.startup)를 앱에 보내고,
        - FastAPI/Starlette가 그 신호를 받아서 여기 등록된 startup 핸들러들을 실행합니다.

        주의
        - 이 함수는 보통 "요청이 오기 전에" 한 번만 실행됩니다.
            (개발 중 `reload=True`면 코드 변경 시 재시작되면서 여러 번 실행될 수 있습니다.)
        """

        # .env 파일에서 환경 변수(예: OPENAI_API_KEY)를 로드합니다.
        # - 벡터 DB를 만드는 OpenAI 임베딩/LLM에서 API 키가 필요할 수 있습니다.
        #dotenv.load_dotenv()

        # 이 함수에서 전역 변수에 값을 "대입"하므로 global 선언이 필요합니다.
        global vector_db

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        logging.info("서버 시작중...")

        # 크로마DB 로드: 기존 DB가 있으면 불러오고, 없으면 새로 만듭니다.
        # (이 작업이 무거우면 서버 시작이 그만큼 늦어질 수 있습니다.)
        vector_db = Utils.load_vector_db()

        logging.info("서버가 정상적으로 가동됩니다.")

        #login("TEST")
        #userchat("TEST", "안녕?")

# "내 서버 IP로 접속하면, 게임(index.html)을 보여줘라"
app.mount("/", StaticFiles(directory="coupang-incident-analysis-agent", html=True), name="Coupang Incident Analysis Agent")

if __name__ == "__main__":

    import uvicorn

    # Uvicorn은 ASGI 서버입니다.
    # - FastAPI는 "웹앱(app)"을 만들어 주는 프레임워크이고
    # - Uvicorn은 그 앱을 네트워크(HTTP)로 실제 서비스하도록 "서버 프로세스"를 실행해주는 런타임입니다.
    #
    # 아래 호출은 내부적으로:
    # 1) main.py 안의 app 객체를 불러오고
    # 2) 지정한 host/port로 소켓을 열어
    # 3) 요청이 들어오면 FastAPI(app)로 전달하고 응답을 반환
    # 을 계속 반복(대기)합니다.
    uvicorn.run(
        # "main:app" 의미:
        # - main: 파이썬 모듈(파일) 이름(main.py)
        # - app : 그 모듈 안에 있는 FastAPI 인스턴스 변수명(app = FastAPI())
        # 즉 "main.py에 있는 app을 서버에 걸어 실행해줘" 라는 뜻입니다.
        "main:app",

        # host="0.0.0.0":
        # - 내 PC의 "모든 네트워크 인터페이스"에서 접속을 받겠다는 의미입니다.
        # - 같은 PC에서만 접속할 거면 127.0.0.1(localhost)도 가능하지만,
        #   Unity가 다른 기기/다른 PC에서 접속해야 한다면 0.0.0.0이 필요할 수 있습니다.
        host="0.0.0.0",

        # port=8101:
        # - 서버가 열어둘 포트 번호입니다.
        # - Unity에서는 예: http://<서버IP>:8101/ping, /message 처럼 호출하게 됩니다.
        port=8101,

        # reload=True:
        # - 개발 편의 기능: 코드가 바뀌면 서버를 자동 재시작합니다.
        # - 개발 중엔 편하지만, 운영(배포) 환경에서는 보통 False로 둡니다.
        reload=True,
    )