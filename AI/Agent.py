from langsmith import traceable
from langgraph.graph import StateGraph, START, END 
from langgraph.checkpoint.memory import MemorySaver
import SystemManager
import GameConfig as config

import AI.Node as Node

class Agent:

    def __init__(self,uid:str):

        """
        Args:
            uid (str): 접속 대상자의 uid. 대화중 기억력에 쓰임.
        """

        self.uid = uid
        self.builder = StateGraph(Node.State)
        self.config = { "configurable" : { "thread_id" : uid } }
        self.memory = MemorySaver()

        """ 그래프에 쓰일 노드들을 정의 함"""
        self.builder.add_node( "question", Node.node_input_question )
        self.builder.add_node( "multi_query", Node.node_multiquery_search )
        self.builder.add_node( "tool_call", Node.node_tool_call )
        self.builder.add_node( "route_next", Node.node_route_next )
        self.builder.add_node( "tools", Node.node_tools )
        self.builder.add_node( "final_answer", Node.node_final_answer )
        self.builder.add_node( "summary", Node.node_summary_conversation )
        self.builder.add_node( "evaluate", Node.node_evaluate ) # 평가용으로 필요시 연결해서 사용

        """ 노드 연결 시작 """
        self.builder.add_edge( START, "question" )
        self.builder.add_edge( "question", "multi_query" )
        self.builder.add_edge( "multi_query", "tool_call" )
        self.builder.add_conditional_edges( 
            "tool_call", 
            Node.node_route_next,
            {
                "need_tools" : "tools",
                "NONE": "final_answer"
            }
        )
        self.builder.add_edge( "tools", "tool_call" )
        self.builder.add_edge( "final_answer", "summary" )
        self.builder.add_edge( "summary", END )

        self.graph = self.builder.compile( checkpointer=self.memory )
        print( self.graph.get_graph().draw_ascii() )

        pass

        
    @traceable
    def run_qa(self, question:str):
        """
        질의응답 실행

            Args:
                question: 사용자의 질문 내용
        """

        if self.graph is None:
            raise Exception("그래프가 초기화되지 않았습니다.")

        return self.graph.invoke( {
            "messages" : [
                ( "user", question )
            ]
        }, config=self.config )

    @traceable
    async def run_qa_ainvoke(self, question:str):
        """
        질의응답 실행

            Args:
                question: 사용자의 질문 내용
        """

        if self.graph is None:
            raise Exception("그래프가 초기화되지 않았습니다.")

        return await self.graph.ainvoke( {
            "messages" : [
                ( "user", question )
            ]
        }, config=self.config )
    
        
    async def run_qa_astream(self, question:str):
        """
        질의응답 비동기 실행 내부 메서드

            Args:
                question: 사용자의 질문 내용
        """

        if self.graph is None:
            raise Exception("그래프가 초기화되지 않았습니다.")

        return self.graph.astream( {
            "messages" : [
                ( "user", question )
            ]
        }, config=self.config )

    async def run_qa_astream_events(self, question: str, version: str = "v2"):
        """
        LangGraph 실행 이벤트 스트림을 반환.

        - userchat_async에서 이벤트를 받아 'final_answer' 노드의 모델 토큰만 골라
          Unity로 스트리밍하기 위한 용도.

        Args:
            question: 사용자 질문
            version: LangGraph 이벤트 포맷 버전(환경에 따라 'v1'/'v2')

        Returns:
            AsyncIterator[dict]: LangGraph 이벤트 스트림
        """

        if self.graph is None:
            raise Exception("그래프가 초기화되지 않았습니다.")

        payload = {
            "messages": [
                ("user", question)
            ]
        }

        # LangGraph 버전에 따라 version 파라미터 지원 여부가 다를 수 있어 폴백 처리
        try:
            return self.graph.astream_events(payload, config=self.config, version=version)
        except TypeError:
            return self.graph.astream_events(payload, config=self.config)