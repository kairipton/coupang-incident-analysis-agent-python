
from langchain_core.tools import tool
from Utils import User
import GameConfig as Config


def phase_process(user: User.Data, is_success:bool) -> str:

    """
    현재 단계 진행 처리

    Args:
        user (User.Data): 사용자 데이터 객체
        is_success (bool): 사용자가 올바른 코드를 입력하여 산소 공급 장치를 복구했는지 여부. 판단은 AI가 함

    Returns:
        페이스 상승시 다음 페이즈의 환영 메세지, 그 외에는 빈 문자열.
    """

    user.phase += 1 if is_success else 0
    user.o2 += -10 if not is_success else 0

    if not is_success:
        return ""

    # 페이즈 전환시에만 제대로 된 환영 메세지 리턴
    return get_welcome_msg( user ) if is_success else ""


def get_welcome_msg(user:User.Data) -> str:
    """
    시작시 환영 메세지 리턴

    Args:
        user (User.Data): 사용자 데이터 객체

    Returns:
        시작 메세지 리턴.
    """
    
    return "환영합니다 엔지니어님."


@tool
def kill_process(pid:int) -> bool:
    """
    프로세스를 종료한다.
    올바른 pid값이 전달 되면 True를 리턴한다.

    Args:
        pid (int): 프로세스 ID

    Returns:
        성공시 True 리턴.
    """

    return pid == 404
    

@tool
def restart_service(service_name:str) -> bool:
    """
    서비스를 재시작한다.
    올바른 서비스 이름이 전달되면 True를 리턴한다.

    Args:
        service_name (str): 서비스 이름

    Returns:
        성공시 True 리턴.
    """
    
    return service_name == "restart_service"


@tool
def clear_cache() -> bool:
    """
    캐시를 비운다.
    언제나 성공적으로 실행 되었다 가정하고 True를 리턴한다.

    Returns:
        True 리턴.
    """
    
    return True

@tool
def rotate_logs() -> bool:
    """
    로그 파일을 압축/정리한다.
    언제나 성공적으로 실행 되었다 가정하고 True를 리턴한다.

    Returns:
        True 리턴.
    """
    
    return True

@tool
def block_ip(ip_address:str) -> bool:
    """
    특정 IP 주소를 차단한다.
    올바른 IP 주소가 전달되면 True를 리턴한다.

    Args:
        ip_address (str): IP 주소

    Returns:
        성공시 True 리턴.
    """
    return ip_address == "192.168.0.50"

@tool
def reset_connection_pool() -> bool:
    """
    연결을 초기화 한다.
    DB 연결 풀(Pool)이 꽉 차서 새로운 연결이 불가능 한 경우 사용할 수 있다.
    언제나 성공적으로 실행 되었다 가정하고 True를 리턴한다.    

    Returns:
        True 
    """
    return True

@tool
def __test_add(a:int, b:int) -> int:
    """
    두 정수를 더한다.

    Args:
        a (int): 첫 번째 정수
        b (int): 두 번째 정수

    Returns:
        int: 두 정수의 합
    """
    return a + b

all_tools = [ kill_process, restart_service, clear_cache, rotate_logs, block_ip, reset_connection_pool, __test_add ]