# Dockerfile: "이미지(Image)를 만드는 레시피(설계도)"
# 이 파일을 기반으로 `docker build`를 하면, FastAPI 앱을 실행할 수 있는 이미지가 만들어집니다.

# 1) 어떤 OS/파이썬 환경을 바탕으로 만들지 선택합니다.
# - python:3.11-slim 은 가벼운 Debian 기반 이미지 + Python 3.11 입니다.
FROM python:3.11-slim

# 2) 파이썬 실행 시 자주 쓰는 기본 설정(선택이지만 흔히 사용)
# - PYTHONDONTWRITEBYTECODE: __pycache__ 같은 바이트코드 파일 생성을 줄임
# - PYTHONUNBUFFERED: print 로그가 즉시 출력되도록 함(도커 로그에서 보기 좋음)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) 컨테이너 안에서 작업할 폴더(프로젝트 루트)를 정합니다.
WORKDIR /app

# 4) 의존성 파일만 먼저 복사해서 설치합니다.
# - 소스코드보다 requirements.txt를 먼저 복사하면, 코드만 바뀔 때는 패키지 설치 캐시를 재사용해서 빌드가 빨라집니다.
COPY requirements.txt ./

# 5) 필요한 파이썬 패키지를 설치합니다.
# - --no-cache-dir: 빌드 결과 용량을 줄임
RUN pip install --no-cache-dir -r requirements.txt

# 6) 소스 코드 및 유니티 빌드 폴더 복사
# - COPY . . 은 현재 경로의 모든 파일을 복사하지만, 
# - 유니티 폴더가 확실히 포함되도록 아래와 같이 명시해주는 것이 안전합니다.
COPY . .
COPY ./deep-space-terminal /app/deep-space-terminal

# 7) (문서용) 이 컨테이너가 사용할 포트를 표시합니다.
# - 실제 포트를 여는 건 docker-compose.yml의 ports 설정입니다.
EXPOSE 8100

# 8) 컨테이너가 "실행"될 때 기본으로 수행할 명령입니다.
# - 여기서는 main.py의 app(FastAPI 인스턴스)을 Uvicorn 서버로 실행합니다.
# - 참고: python main.py로 실행해도 되지만, 컨테이너에선 보통 uvicorn 명령으로 직접 실행합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
