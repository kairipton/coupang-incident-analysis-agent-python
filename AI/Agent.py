from langsmith import traceable
from langchain_openai import ChatOpenAI
import langchain_core.prompts as Prompt
import langchain_core.runnables as Runnable
import langchain_core.chat_history as History
from langchain_community.chat_message_histories import ChatMessageHistory
import langchain_core.vectorstores as VectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from pydantic import BaseModel, Field
import GameConfig as config
import AI.Tools as Tools

class GameResponse(BaseModel):
    """
    JsonOutputParser를 쓸때 사용 하던 레거시 코드.
    스트리밍에 적합하지 않아 지금은 사용하지 않음.
    """
    answer: str = Field( description="플레이어에게 전달할 답변 내용 (힌트 포함)" )
    is_success:bool = Field( description="플레이어가 올바른 코드를 입력하여 산소 공급 장치를 복구했는지 여부" )
    

class Agent:

    # 질문 변환 맵
    question_transform_map = ["주인공 -> 플레이어", "피 -> 체력", "달면 -> 떨어지면"]

    def __init__(self, retriever:VectorStore.VectorStoreRetriever, session_id:str, get_session_history_func:History.BaseChatMessageHistory):
        """
        Args:
            retriever: 문서 검색기 지정
            session_id: 세션 ID 지정
            get_session_history_func: 세션 ID로 대화 기록을 가져오는 함수 지정
        """

        # LLM
        self.llm = ChatOpenAI( model=config.llm_model_name )

        # 리트리버
        self.retriever: VectorStore.VectorStoreRetriever = retriever

        # 대화 히스토리 변수
        self.session_id: str = session_id
        self.get_session_history: History.BaseChatMessageHistory = get_session_history_func

        # Json규격 답변으로 고정 시킬때.
        # 생성자를 직접 쓰고 변수들을 직접 정한다.
        self.qa_prompt = Prompt.PromptTemplate.from_template("""


            {input_result_prompt}

            # [CURRENT STATUS]
            - 산소 잔여량: {o2}
            - 현재 진행 단계: {phase}

            # [MEMORY DATA]
            **[Phase {phase}] 정보만 사용하십시오.**
            {document}

            [이전 기록]: {history}
            [엔지니어 입력]: {question}

            """,
        )



    def set_retriever(self, retriever:VectorStore.VectorStoreRetriever) -> None:
        """
        리트리버 설정 및 변경

        Args:
            retriever: 문서 검색기 지정
        """
        self.retriever = retriever

    def set_session_history(self, session_id:str, get_session_history_func:History.BaseChatMessageHistory) -> None:
        """
        세션 대화 기록 함수 설정 및 변경

        Args:
            session_id: 세션 ID 지정
            get_session_history_func: 세션 ID로 대화 기록을 가져오는 함수 지정
        """
        self.session_id = session_id
        self.get_session_history = get_session_history_func

    @traceable
    def run_qa(self, question:str, user_var:dict, run_type:int=0):
        """
        질의응답 실행

            Args:
                question: 사용자의 질문 내용
                user_var: 사용자의 변수 정보 딕셔너리
                run_type: 질의 실행 방법 설정 (0: invoke, 1: stream)

            Returns:
                스트림 가능한 답변 객체
        """

        if self.session_id == "":
            raise Exception("세션 ID가 설정되지 않았습니다. set_session_history()를 먼저 호출하세요.")
        
        if self.get_session_history is None:
            raise Exception("대화 기록 함수가 설정되지 않았습니다. set_session_history()를 먼저 호출하세요.")
        
        if self.retriever is None:
            raise Exception("리트리버가 설정되지 않았습니다. set_retriever()를 먼저 호출하세요.")
 
        # 체인 만들고
        rag_chain = Runnable.RunnableWithMessageHistory(
            runnable = self.final_chain, # 최종 체인 로직이 여기에...
            get_session_history = self.get_session_history, # 대화 내역을 가져오는 메서드
            input_messages_key = "question",
            history_messages_key = "history"
        )

        # 호출!
        return self.__run_qa_chain( question, user_var, rag_chain, run_type )

    @traceable
    async def run_qa_async(self, question:str, user_var:dict):
        """
        질의응답 실행

            Args:
                question: 사용자의 질문 내용
                user_var: 사용자의 변수 정보 딕셔너리

            Returns:
                스트림 가능한 답변 객체
        """

        if self.session_id == "":
            raise Exception("세션 ID가 설정되지 않았습니다. set_session_history()를 먼저 호출하세요.")
        
        if self.get_session_history is None:
            raise Exception("대화 기록 함수가 설정되지 않았습니다. set_session_history()를 먼저 호출하세요.")
        
        if self.retriever is None:
            raise Exception("리트리버가 설정되지 않았습니다. set_retriever()를 먼저 호출하세요.")
 
        # 체인 만들고
        rag_chain = Runnable.RunnableWithMessageHistory(
            runnable = self.final_chain, # 최종 체인 로직이 여기에...
            get_session_history = self.get_session_history, # 대화 내역을 가져오는 메서드
            input_messages_key = "question",
            history_messages_key = "history"
        )

        # 호출!
        return await self.__run_qa_chain_async( question, user_var, rag_chain )
    
    def __run_qa_chain(self, question:str, user_var:dict, rag_chain:Runnable.RunnableWithMessageHistory, run_type:int):
        """
        질의응답 실행 내부 메서드
        """
        if run_type == 0:
            return rag_chain.invoke(
                { "question": question, "user_var" : user_var },

                # configurable은 랭체인에서 쓰는 예약처럼 쓰이고, 여러가지 값이 있음.
                # session_id도 configurable 내부에서 예약어 처럼 쓰이며, 이거는 RunnableWithMessageHistory 사용시 파라미터로 명시하여 변경 가능.
                # RunnableWithMessageHistory에 session_id 전달. 
                config={ "configurable": { "session_id" : self.session_id } } 
            )
        
        elif run_type == 1:
            return rag_chain.stream(
                { "question": question, "user_var" : user_var },
                config={ "configurable": { "session_id" : self.session_id } } 
            )
        
        else:
            raise Exception( f"알 수 없는 run_type 값입니다: {run_type}" )

        
    async def __run_qa_chain_async(self, question:str, user_var:dict, rag_chain:Runnable.RunnableWithMessageHistory):
        """
        질의응답 비동기 실행 내부 메서드
        """
        return rag_chain.astream(
            { "question": question, "user_var" : user_var },
            config={ "configurable": { "session_id" : self.session_id } } 
        )
        

    @staticmethod
    def __format_docs( docs ):
        """
        리트리버로가 검색하여 가져온 문서들을 문자열 뭉탱이로 리턴함
        """
        return "\n\n".join([doc.page_content for doc in docs])