
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from Utils import User
import config as Config


# @tool
# def __test_add(a:int, b:int) -> int:
#     """
#     두 정수를 더한다.

#     Args:
#         a (int): 첫 번째 정수
#         b (int): 두 번째 정수

#     Returns:
#         int: 두 정수의 합
#     """
#     return a + b


@tool
def get_web_search(query:str, k:int=3) -> list[dict[str, str | float]]:
    """
    웹 검색 실행

    Args:
        query(str): 검색어 입력
        k(int): 검색 결과중 가장 관련성이 높은 k개만 가져 오겠다는 의미.

    Returns:
        list[dict[str, str | float]]: 검색 결과의 리스트, 각 결과는 제목, URL, 내용, 점수를 포함
            - title(str): 검색된 컨텐츠의 제목
            - url(str): 검색된 컨텐츠의 URL
            - content(str): 검색된 컨텐츠의 내용
            - score(float): 검색 결과의 점수
    """
    search = TavilySearchResults(k=k)
    search_results = search.run(query)

    titles = []
    urls = []
    contents = []
    scores = []

    for r in search_results:
        titles.append( r.get("title", "") )
        urls.append( r.get("url", "") )
        contents.append( r.get("content", "") )
        scores.append( r.get("score", 0.0) )

    all_results = list( zip( titles, urls, contents, scores ) )
    result: list[dict[str, str | float]] = []
    for (title, url, content, score) in all_results:
        if score >= 0.7:
            result.append( {
                "title" : title,
                "url" : url,
                "content" : content,
                "score" : score
            } )

    return result

@tool
def calculate_reporting_delay(start_time:str, report_time:str):
    """
    사고 인지 시간(start_time)과 신고 시점(report_time) 계산하여 신고 지연 시간을 계산하는 도구입니다.
    법적 위반 여부를 판단할 떄 사용할 수 있습니다.

    Args:
        start_time(str): 사고 인지 시간 (예: "2025-11-17T16:00:00Z")
        report_time(str): 신고 시점 (예: 예: "2025-11-19 21:35")

    Returns:
        dict: 신고 지연 시간과 법적 위반 여부를 포함하는 딕셔너리
            - delay_hours(float): 사고 인지와 신고 시점 사이의 지연 시간(시간 단위)
            - is_violation(bool): 지연 시간이 법적 허용 범위를 초과하는지 여부
            - legal_limit_hours(float): 법적으로 허용되는 최대 지연 시간(시간 단위)
    """

    from datetime import datetime, timezone
    time_format = "%Y-%m-%d %H:%M"
    start = datetime.strptime( start_time, time_format )
    end = datetime.strptime( report_time, time_format )

    delay = end - start
    total_hours = delay.total_seconds() / 3600
    limit_hours = 24
    is_violation = total_hours > limit_hours

    return {
        "delay_hours" : total_hours,
        "is_violation" : is_violation,
        "legal_limit_hours" : limit_hours,
    }

@tool
def total_leak_stats(member_base: int, additional_member: int, shipping_views: int, order_views: int):
    """
    정부 기관(과기정통부, 개인정보보호위원회)별로 파편화된 유출 및 노출 수치를 통합하여 전체 피해 통계를 산출합니다.

    Args:
        member_base (int): 과기정통부가 발표한 '내정보 수정 페이지' 유출 계정 수 (예: 33673817).
        additional_member (int): 개보위에 추가로 신고된 회원 계정 수 (예: 165455).
        shipping_views (int): 성명, 주소 등이 포함된 '배송지 목록 페이지'의 총 조회 횟수 (예: 148056502).
        order_views (int): '주문목록 페이지'의 총 조회 횟수 (예: 100000).

    Returns:
        dict: 확정 유출 계정 합계 및 페이지 노출 통계를 구조화한 데이터.
    """
    total_member_leaks = member_base + additional_member
    total_page_views = shipping_views + order_views
    
    return {
        "total_confirmed_member_leaks": total_member_leaks,
        "exposure_stats": {
            "total_views": total_page_views,
            "shipping_list_views": shipping_views,
            "order_list_views": order_views
        }
    }

@tool
def audit_technical_vulnerability(vulnerability_list: list):
    """
    쿠팡 침해사고의 기술적 원인들을 분석하여 보안 영역별 결함 보고서를 작성합니다.

    Args:
        vulnerability_list (list): 문서에서 추출된 취약점 키워드 리스트. 
                                  예: ["서명키 미갱신", "전자 출입증 위변조", "모의해킹 취약점 방치"]
    
    Returns:
        dict: 인증, 계정관리, 운영 보안별 결함 상세 및 재발 방지 대책 매핑 결과.
    """
    audit_report = {
        "authentication_security": [],  # 인증 체계 관련
        "identity_management": [],      # 퇴사자 및 계정 관리 관련
        "security_governance": []       # 모의해킹 등 프로세스 관련
    }
    
    for vuln in vulnerability_list:
        if "출입증" in vuln or "인증" in vuln:
            audit_report["authentication_security"].append({
                "issue": vuln,
                "detail": "정상 절차를 거치지 않은 위·변조 전자 출입증 검증 미흡",
                "remedy": "전자 출입증 탐지 및 차단 체계 도입"
            })
        elif "서명키" in vuln or "퇴사" in vuln:
            audit_report["identity_management"].append({
                "issue": vuln,
                "detail": "개발자 퇴사 후 서명키 즉시 갱신 미이행",
                "remedy": "서명키 발급·폐기 관리 체계 강화"
            })
        elif "모의해킹" in vuln or "취약점" in vuln:
            audit_report["security_governance"].append({
                "issue": vuln,
                "detail": "기존 모의해킹으로 발견된 취약점 조치 미흡",
                "remedy": "취약점 조치 여부 정기 점검 및 관리 강화"
            })
            
    return audit_report


@tool
def check_compliance_status(compliance_items: list):
    """
    정부 기관의 주요 명령 및 권고 사항에 대한 쿠팡의 이행 상태와 법적 위반 정황을 추적합니다.

    Args:
        compliance_items (list): 추적할 법적 명령 또는 권고 항목 리스트.
                                예: ["자료보전 명령", "유출 재통지", "조사 자료 제출"]
    
    Returns:
        dict: 각 항목별 이행 상태(준수/위반/주의) 및 관련 법적 근거와 상세 사유.
    """
    status_report = {}
    
    for item in compliance_items:
        if "자료보전" in item or "로그" in item:
            status_report[item] = {
                "status": "VIOLATION",
                "detail": "자료보전 명령 이후에도 웹 및 앱 접속기록을 삭제함",
                "action": "수사 의뢰 진행 중"
            }
        elif "재통지" in item or "노출" in item:
            status_report[item] = {
                "status": "CORRECTED",
                "detail": "'노출' 통지를 '유출'로 수정하고 누락된 항목을 포함하여 재통지함",
                "action": "7일 이내 조치 결과 제출 명령"
            }
        elif "자료 제출" in item or "협조" in item:
            status_report[item] = {
                "status": "WARNING",
                "detail": "자료 제출 거부 및 지연 제출 반복으로 조사 방해 행위 판단",
                "action": "제재 처분 시 가중요건 적용 경고"
            }
        else:
            status_report[item] = {
                "status": "UNDER_INVESTIGATION",
                "detail": "개인정보보호법 위반 여부 등에 대해 현재 조사 진행 중",
                "action": "관계 부처 합동 조사 진행"
            }
            
    return status_report


all_tools = [ get_web_search, calculate_reporting_delay, total_leak_stats, audit_technical_vulnerability, check_compliance_status ]