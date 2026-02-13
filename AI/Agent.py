from langsmith import traceable
from langchain_openai import ChatOpenAI
import langchain_core.prompts as Prompt
import langchain_core.runnables as Runnable
import langchain_core.chat_history as History
from langchain_community.chat_message_histories import ChatMessageHistory
import langchain_core.vectorstores as VectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_agent
from pydantic import BaseModel, Field
import SystemManager
import GameConfig as config

class Agent:

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


        
        self.agent_prompt = """
            당신은 시스템 관리자를 돕는 AI Agent 입니다.
            사용자의 질문을 토대로 현재 상황을 해결할 수 있는 도움을 주십시오.
            상황을 해결하기 위한 방법을 사용자에게 알려주고,
            필요하면 tools를 호출해서 상황을 직접 해결하고 사용자에게 결과를 알려주십시오.

            사용자의 질문: {question}
        """

        # 여기에 넣는 프롬프트는 create_agent할때 고정이다.
        # 다시 create_agent를 호출하기 전까지는 고정.
        self.agent = create_agent(
            model = self.llm,
            tools = SystemManager.all_tools,
            system_prompt=self.agent_prompt
        )

        self.rag_chain = (
            Runnable.Runnable
        )

        
    @traceable
    def run_qa(self, question:str):
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

        # 호출!
        result = self.agent.invoke( {
            "messages": [
                ("user", question)
            ]
        })

        # # create_agent는 보통 {"messages": [...]} 형태로 반환합니다.
        # if isinstance(result, dict) and "messages" in result and result["messages"]:
        #     last_msg = result["messages"][-1]
        #     return getattr(last_msg, "content", str(last_msg))

        # return str(result)
        return result

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
 
        # # 체인 만들고
        # rag_chain = Runnable.RunnableWithMessageHistory(
        #     runnable = self.final_chain, # 최종 체인 로직이 여기에...
        #     get_session_history = self.get_session_history, # 대화 내역을 가져오는 메서드
        #     input_messages_key = "question",
        #     history_messages_key = "history"
        # )

        # 호출!
        #return await self.__run_qa_chain_async( question, user_var, rag_chain )
        return await self.agent.ainvoke( question )
    
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