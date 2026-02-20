import os
import sys
import dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import GameConfig as Config
import SystemManager
import Node
from Node import llm_with_tool



builder = StateGraph(Node.State)

# region 노드 정의
""" 그래프에 쓰일 노드들을 정의 함"""
builder.add_node( "question", Node.node_input_question )
builder.add_node( "multi_query", Node.node_multiquery_search )
#builder.add_node( "search", Node.node_hybrid_search )
builder.add_node( "run_qa", Node.node_run_qa )
builder.add_node( "route_next", Node.node_route_next )
builder.add_node( "tools", Node.node_tools )
builder.add_node( "summary", Node.node_summary_conversation )
builder.add_node( "evaluate", Node.node_evaluate )
# endregion


# region 그래프 정의
""" 노드 연결 시작 """
builder.add_edge( START, "question" )
builder.add_edge( "question", "multi_query" )
builder.add_edge( "multi_query", "run_qa" )
builder.add_conditional_edges( 
    "run_qa", 
    Node.node_route_next,
    {
        "need_tools" : "tools",
        "NONE": "summary"
    }
)
builder.add_edge( "tools", "run_qa" )
builder.add_edge( "summary", END )
# endregion

memory = MemorySaver()
graph = builder.compile( checkpointer=memory )
print( graph.get_graph().draw_ascii() )


config = { "configurable" : { "thread_id" : "test_uid" } }

question = "고객 정보가 얼마나 유출 됐나요?"
result = graph.invoke({ 
    "messages" : [ 
        ( "user", question ) 
    ]
}, config=config )

# 0은 사용자의 입력, -1이 대체로 AI의 답변이지만, 도구사용을 마지막으로 했을 경우 메세지가 아닐 수 있다.
print( result["messages"][0] )
print( result["messages"][-1] )

summary = result.get( "summary", "" )
print( "\n[대화 요약]" )
print( summary )

