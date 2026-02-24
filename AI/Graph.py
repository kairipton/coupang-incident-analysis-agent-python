import os
import sys
import dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import Node



builder = StateGraph(Node.State)

# region 노드 정의
""" 그래프에 쓰일 노드들을 정의 함"""
builder.add_node( "question", Node.node_input_question )
builder.add_node(
    "multi_query",
    Node.node_multiquery_search,
    # 노드별 커스텀 메타데이터 예시
    # - astream_events()로 나오는 ev['metadata']에 함께 포함될 수 있음
    # - Unity 진행 UI에 쓸 "표시 문구/단계" 같은 값을 여기에 넣어두면 편함
    metadata={
        "unity_label": "멀티 쿼리 진행중",
    },
)
builder.add_node( "tool_call", Node.node_tool_call )
builder.add_node( "route_next", Node.node_route_next )
builder.add_node( "tools", Node.node_tools )
builder.add_node( "final_answer", Node.node_final_answer )
builder.add_node( "summary", Node.node_summary_conversation )
builder.add_node( "evaluate", Node.node_evaluate )
# endregion


# region 그래프 정의
""" 노드 연결 시작 """
builder.add_edge( START, "question" )
builder.add_edge( "question", "multi_query" )
builder.add_edge( "multi_query", "tool_call" )
builder.add_conditional_edges( 
    "tool_call", 
    Node.node_route_next,
    {
        "need_tools" : "tools",
        "NONE": "final_answer"
    }
)
builder.add_edge( "tools", "tool_call" )
builder.add_edge( "final_answer", "evaluate" )
builder.add_edge( "evaluate", "summary" )
builder.add_edge( "summary", END )
# endregion

memory = MemorySaver()
graph = builder.compile( checkpointer=memory )
print( graph.get_graph().draw_ascii() )


config = { "configurable" : { "thread_id" : "test_uid" } }


while True:
    question = input( "질문: ")
    result = graph.invoke({ 
        "messages" : [ 
            ( "user", question ) 
        ]
    }, config=config )

    # 0은 사용자의 입력, -1이 대체로 AI의 답변이지만, 도구사용을 마지막으로 했을 경우 메세지가 아닐 수 있다.
    #print( result["messages"][0] )
    #print( result["messages"][-1] )
    pprint( result )

    # summary = result.get( "summary", "" )
    # print( "\n[대화 요약]" )
    # print( summary )

