from Utils import User
import GameConfig as Config

class GameManager:
    """
    게임 진행 처리 클래스.
    """

    def __init__(self):
        pass

    @staticmethod
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
        return GameManager.get_welcome_msg( user ) if is_success else ""

    @staticmethod
    def get_welcome_msg(user:User.Data) -> str:
        """
        현재 페이즈의 환영 메세지 리턴

        Args:
            user (User.Data): 사용자 데이터 객체

        Returns:
            현재 페이즈의 환영 메세지
        """
        
        #msg_len = len(Config.phase_welcome_messages)
        msg_len = len( Config.phase_data )

        idx = user.phase-1
        welcome_msg = ""
        if idx <= msg_len -1:
            #welcome_msg = Config.phase_welcome_messages[ user.phase-1 ]
            welcome_msg = Config.phase_data[ user.phase-1 ]["msg"]
        
        return welcome_msg