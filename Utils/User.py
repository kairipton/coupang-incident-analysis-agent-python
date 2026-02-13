from langchain_community.chat_message_histories import ChatMessageHistory
from datetime import datetime, timedelta

class Data:
    """사용자 상태와 대화 기록을 보관하는 객체.

    Attributes:
        uid (str): 사용자 ID. 초기값은 -1.
        history (ChatMessageHistory): 사용자 대화 기록.
        cpu (int): CPU 상태. 0은 정상, 1은 비정상 상태.
        memory (int): 메모리 사용량. 0은 정상, 1은 비정상 상태.
        network (int): 네트워크 사용량. 0은 정상, 1은 비정상 상태.
    """
    
    def __init__(self, uid: str):
        self.uid: str = uid
        self.history = ChatMessageHistory()

        # 아래 값부터 0은 정상, 1은 과부하와 같은 비정상 상태로 취급함
        self.cpu = 0
        self.memory = 0
        self.network = 0

class DB:
    """
    사용자 대화 기록과 정보를 보관.

    Attributes:
        _user_store (dict): 사용자 ID를 키로 하고 ChatMessageHistory 객체를 값으로 하는 딕셔너리.
        _user_expire_min (int): 사용자 정보 만료 시간(분).
    """

    _user_store: dict[str, Data] = {}

    _user_expire_min: int = 15

    @staticmethod
    def get_or_make_user(user_id:str) -> Data:
        """
        유저 정보를 가져옴

        Args:
            user_id: 플레이어 uid

        Returns:
            BaseChatMessageHistory: 대화 기록 객체
        """
        if user_id not in DB._user_store:
            DB._user_store[user_id] = Data( user_id)

        DB.check_expire_users()

        return DB._user_store[ user_id ]
    
    def delete_user(user_id:str) -> None:
        """
        유저 정보 삭제
        
        Args:
            user_id: 플레이어 uid
        """
        if user_id in DB._user_store:
            del DB._user_store[user_id]
    
    def get_history(user_id:str) -> ChatMessageHistory:
        """
        특정 사용자의 대화 내역을 가져옴.
        랭체인과 직접 연결되는 메서드.
        """

        if user_id not in DB._user_store:
            raise Exception("해당 유저 ID의 사용자가 존재하지 않습니다.")
        
        DB.check_expire_users()
        
        return DB._user_store[user_id].history
    
    @staticmethod
    def check_expire_users():
        """
        마지막 로그인 시간 기준으로 일정 시간이 지난 유저들은 파기 하도록 함
        """
        
        now = datetime.now()
        expire_delta = timedelta( minutes=DB._user_expire_min )
        uids = []
        for user_key in DB._user_store:
            u = DB._user_store[user_key]

            if now - u.last_login >= expire_delta:
                uids.append( u.uid )
        
        for uid in uids:
            print( f"{uid} 유저는 파기됩니다" )
            del DB._user_store[uid]
