from langchain.agents import create_agent
from langchain_core.tools import tool


@tool
def kill_process(pid:int) -> None:
    """
    프로세스를 종료한다.

    Args:
        pid (int): 프로세스 ID

    Returns:
        None
    """
    pass

@tool
def restart_service(service_name:str) -> None:
    """
    서비스를 재시작한다.

    Args:
        service_name (str): 서비스 이름

    Returns:
        None
    """
    pass


@tool
def clear_cache() -> None:
    """
    캐시를 비운다.

    Returns:
        None
    """
    pass

@tool
def rotate_logs() -> None:
    """
    로그 파일을 압축/정리한다.

    Returns:
        None
    """
    pass

@tool
def block_ip(ip_address:str) -> None:
    """
    특정 IP 주소를 차단한다.

    Args:
        ip_address (str): IP 주소

    Returns:
        None
    """
    pass

@tool
def reset_connection_pool() -> None:
    """
    연결을 초기화 한다.
    DB 연결 풀(Pool)이 꽉 차서 새로운 연결이 불가능 한 경우 사용할 수 있다.

    Returns:
        None
    """
    pass

all_tools = [ kill_process, restart_service, clear_cache, rotate_logs, block_ip, reset_connection_pool ]