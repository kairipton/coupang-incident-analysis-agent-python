import os
from langsmith import traceable

# 1. 코드에서 강제로 설정 (환경 변수 파일 무시)
# 여기에 본인의 키를 직접 붙여넣으세요.
MY_API_KEY = "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"  # <--- 여기에 실제 키 입력

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = MY_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Deep_Space_Terminal"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = MY_API_KEY
os.environ["LANGSMITH_PROJECT"] = "Deep_Space_Terminal"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# 2. 테스트 함수 실행
@traceable(name="Direct_Test")
def test_connection():
    return "Connection Success!"

print("--- LangSmith 연결 테스트 중 ---")
try:
    # 실행하는 순간 전송됨
    result = test_connection() 
    print(f"✅ 실행 완료: {result}")
    print("👉 대시보드(Deep_Space_Terminal 프로젝트)를 새로고침 해보세요!")
except Exception as e:
    print(f"❌ 연결 실패: {e}")