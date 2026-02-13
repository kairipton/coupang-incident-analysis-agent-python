import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import dotenv
from typing import Annotated, TypedDict, NotRequired
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

import Utils.Utils as Utils
import GameConfig as Config
from SystemManager import all_tools

dotenv.load_dotenv()

PROJECT_ROOT = Utils.find_project_root( __file__ )

# 도구까지 바인딩 시켜놓음
llm = ChatOpenAI( model=Config.llm_model_name )
llm_with_tool = llm.bind_tools( all_tools )


class State(TypedDict):
    question: Annotated[str, "question"] # 원본 질문. 그래프 루프가 시작할때 입력됨.
    multi_questions: Annotated[list, "multi_questions"] # 멀티 리트리버로 가져온 질문들.
    documents: Annotated[list, "documents"] # 하이브리드 서치로 가져온 문서들.
    messages: Annotated[list, add_messages] # 대화 히스토리
    summary: Annotated[str, "summary"] # 대화 요약본. 응답 후 마지막에 갱신 됨.

# region 유틸리티
def __format_messages_for_summary(messages:list) -> str:
    """
    LLM과의 대화 내역을 읽기 쉬운 형태로 변환

    Args:
        messages (list): 대화 내역 메시지 리스트

    Returns:
        읽기 쉬운 형태로 변환된 문자열
    """
    formatted_list = []
    
    for msg in messages:
        # 1. 역할 설정
        if isinstance(msg, HumanMessage):
            role = "사용자"
            content = msg.content
        elif isinstance(msg, SystemMessage):
            role = "시스템"
            content = msg.content
        elif isinstance(msg, ToolMessage):
            role = "도구 결과"
            content = f"실행 결과: {msg.content}"
        elif isinstance(msg, AIMessage):
            role = "AI"
            content = msg.content
            
            # AI가 도구 호출을 시도한 경우 정보 추가
            if msg.tool_calls:
                tool_info = []
                for tool in msg.tool_calls:
                    # C#의 Dictionary 처럼 tool['name'], tool['args']에 접근
                    t_name = tool['name']
                    t_args = json.dumps(tool['args'], ensure_ascii=False)
                    tool_info.append(f"[도구 호출: {t_name}({t_args})]")
                
                # 기존 답변 내용 뒤에 도구 호출 정보를 붙여줌
                content += " " + " ".join(tool_info)
        else:
            continue # 다른 메시지 타입은 무시
            
        formatted_list.append(f"{role}: {content}")
    
    return "\n".join(formatted_list)
# endregion

# region 노드 정의

""" 툴 노드. LLM이 알아서 실행하게 된다. """
node_tools = ToolNode( all_tools )

def node_input_question(state: State):
    """
    사용자의 질문을 입력받는 노드.
    현재 대화를 시작할때의 원본 질문을 저장해둔다.

    Returns:
        "question": 사용자의 질문
    """

    print( f"질문: {state['messages'][-1].content}" ) 

    return { "question": state["messages"][-1].content }

def node_run_qa(state: State):
    """
    LLM에게 질문을 던지고 답변을 받는 노드.
    Multi-Query Retriever로 얻은 질문들과, Hybrid Search로 얻은 문서들이 필요함.

    LLM의 출력은 크게 두 가지 중 하나인데, 사용자가 제어하는게 아님.
    1) 도구가 필요 없으면: 그냥 자연어 답변(content)을 가진 AIMessage
    2) 도구가 필요하면: tool_calls가 들어있는 AIMessage

    Returns:
        "messages": LLM의 답변 메시지
    """

    prompt = f"""
        당신은 "시스템 엔지니어" 를 도와주는 AI 어시스턴트 입니다.
        "시스템 엔지니어" 는 사용자를 말합니다.
        사용자는 현재 발생한 문제를 해결하기 위해 당신에게 [질문]합니다.
        주어진 [정보]를 바탕으로 사용자에게 대답해주세요.

        [질문]
        {state["multi_questions"]}

        [정보] 
        {state["documents"]}
        
        [대화 내역]
        {state["messages"]}
    """


    answer = llm_with_tool.invoke( prompt )
    return { "messages" : [answer] }


def node_summary_conversation(state: State):
    """
    대화 내역 요약.
    최종 답을 내리기 전 현재까지의 대화 내역을 요약한다.

    Returns:
        "summary": 요약된 대화 내용
    """

    summary = state.get( "summary", "" )
    messages = __format_messages_for_summary( state.get( "messages", [] ) )

    prompt = f"""
        아래는 시스템 엔지니어(사용자)와 AI 어시스턴트(당신) 간의 최신 [대화 내역]과 이전 대화 내역의 [요약본] 입니다.
        [대화 내역]과 [요약본] 을 보고 하나의 대화로 요약 해주세요.
        출력시 대화 요약된 대화 내역만 출력 하세요.
        
        [요약본]
        {summary}

        [대화 내역]
        {messages}
    """

    answer = llm.invoke( prompt )
    return { "summary": getattr( answer, "content", str(answer) ) }


def node_route_next(state:State):
    """
    가장 마지막 메세지에 tool_calls가 있으면 need_tools를, 없으면 "NONE"을 리턴.

    Returns:
        툴 사용이 필요하다면 "need_tools"를, 아니라면 "NONE"을 리턴.
    """

    msgs = state.get( "messages", [] )
    last = msgs[-1] if msgs else None
    tool_calls = getattr( last, "tool_calls", None )

    return "need_tools" if tool_calls else "NONE"


def node_multiquery(state:State):
    """
    사용자의 질문을 여러개의 질문으로 변환.

    Returns:
        "multi_questions" 에 여러개의 질문을 넣음
    """

    query = f"""
    아래의 질문을 4개의 쿼리로 바꾸세요.
    각 쿼리는 한줄씩, 쿼리만 출력하세요.

    질문: {state["question"]}
    """
    
    answer = llm_with_tool.invoke( query )
    queries = getattr( answer, "content", str(answer) )
    multi_questions = [ state["question"] ] + queries.split( "\n" ) # extend 메서드 써도 됨 (메모리 효율)

    return { "multi_questions": "\n".join( multi_questions ) }


def node_hybrid_search(state:State):
    """
    여러개의 질문을 바탕으로 문서 검색.

    Returns:
        "documents" 에 검색된 문서들을 넣음
    """
    k = 6

    # region BM25 리트리버 준비
    doc_path = PROJECT_ROOT / "Knowledge Base" / "tech_manual.md"
    loader = TextLoader(str(doc_path), encoding="utf8")
    docs = loader.load()

    spliiter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200 )
    chunks = spliiter.split_documents( docs )

    for c in chunks:
        c.metadata[ "category" ] = "manual"

    bm25_retriever = BM25Retriever.from_documents( chunks, k=k )
    # endregion

    # region Vector 리트리버 준비
    db = Utils.load_vector_db()
    vector_retriever = db.as_retriever( k=k )
    # endregion

    # 하이브리드(앙상블) 리트리버
    hybrid_retriever = EnsembleRetriever(
        retrievers = [ bm25_retriever, vector_retriever ],
        weights = [ 0.5, 0.5 ],
    )

    docs = hybrid_retriever.invoke( state["multi_questions"] )
    doc_contents = [d.page_content for d in docs]
    
    return { "documents": doc_contents } 


# endregion

state = node_multiquery( State( question="계란 후라이 하는 방법?" ) )
node_hybrid_search( state )

