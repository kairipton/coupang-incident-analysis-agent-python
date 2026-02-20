import dotenv
dotenv.load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults

# 도구 초기화
# k=3은 검색 결과 중 가장 관련성 높은 3개만 가져오겠다는 의미입니다.
search = TavilySearchResults(k=3)

# 직접 실행 테스트
results = search.run("2026년 현재 대한민국에서 가장 인기 있는 프로그래밍 언어는?")
print(results)