import os
import sys
import json
from pathlib import Path

# `@lru_cache` 데코레이터 설명
# - 파이썬의 함수 결과 캐시(메모이제이션) 기능입니다.
# - 같은 인자로 함수를 다시 부르면, 내부 계산을 다시 하지 않고 이전 결과를 그대로 돌려줍니다.
# - 여기서는 "리랭커 모델 로딩"을 매 호출마다 하지 않게 하려고 사용합니다.
from functools import lru_cache
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
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

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
    documents: Annotated[list, "documents"] # 멀티쿼리 리트리버로 가져온 문서들.
    messages: Annotated[list, add_messages] # 대화 히스토리
    summary: Annotated[str, "summary"] # 대화 요약본. 응답 후 마지막에 갱신 됨.
    reference: NotRequired[Annotated[str, "reference answer"]] # 평가용 정답. 라이브에서는 안쓰임.

# region 유틸리티
def __get_hybrid_retriever(k=6) -> EnsembleRetriever:
    """
    BM25 + Vector 결과를 rank fusion(Weighted RRF)으로 결합한 하이브리드 리트리버를 반환하는 유틸 함수.

    Args:
        k (int): 각 리트리버에서 반환할 문서 수

    Returns:
        EnsembleRetriever: 하이브리드 리트리버 객체
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
    return EnsembleRetriever(
        retrievers = [ bm25_retriever, vector_retriever ],
        weights = [ 0.5, 0.5 ],
    )

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

@lru_cache(maxsize=1)
def __get_cross_encoder_reranker():

    """
    Cross-Encoder 리랭커 모델을 1회만 로딩해서 재사용.
    """

    """
    Cross-Encoder는 (질문, 문서) 쌍을 **함께** 입력으로 받아서 점수를 직접 예측합니다.
    즉 "질문과 문서를 각각 임베딩"해서 유사도를 보는 방식(bi-encoder)과 다르게,
    두 텍스트 사이의 상호작용을 모델이 바로 보면서 점수화하므로 리랭킹 품질이 잘 나오는 편입니다.

    주의: 처음 로딩 시 모델 다운로드/로딩 비용이 있고, 문서 개수만큼 쌍을 점수화하므로
    후보 문서 수가 많을수록 시간이 늘어납니다.
    """
    model_name = Config.cross_encoder_rerank_model_name

    device = Config.cross_encoder_device

    return CrossEncoder(model_name, device=device)
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


def node_multiquery_search(state:State):
    """
    사용자의 질문을 여러개의 질문으로 변환(Multi-Query)하여, 
    앙상블 리트리버(Hybrid Search)를 이용해 RAG 검색 결과를 가져옴

    Returns:
        "multi_questions" 에 여러개의 질문을 넣음
    """
    
    # 하이브리드 서치로 잘 가져올 수 있도록,
    # 질문을 잘 나눌수 있게 멀티쿼리를 위한 프롬프트를 작성 해야 함.
    template = PromptTemplate.from_template( """
        사용자의 질문을 분석하여, 지식 베이스(매뉴얼)에서 정보를 가장 잘 찾을 수 있도록 검색 엔진용 쿼리를 5개 생성하세요. 
        각 쿼리는 서로 다른 키워드와 관점을 포함해야 합니다.
        각 쿼리는 한줄씩, 쿼리만 출력하세요.
                                 
        질문: {question}
    """)

    # MultiQueryReriever는 멀티 쿼리와 리트리버를 이용한 검색을 한큐에 해버린다.
    # 이때 리트리트가 앙상블 리트리버라면 하이브리드 개념상 하이브리드 서치이므로,
    # Multi Query + Hybrid Search가 한큐에 되는 셈.
    # 다만, 멀티 쿼리로 질문이 5개가 된 경우, k가 6개라면, 질문당 문서를 6개를 가져오게 되는데,
    # 5 * 6 = 최대 30개의 문서가 검색될 수 있고, 이 문서를 전부 프롬프트로 날리면 토큰 사용량이 너무 커진다.
    # 그러므로 검색 후 적절히 걸러주는게 좋다. (Rerank 등..)
    mqr = MultiQueryRetriever.from_llm(
        retriever=__get_hybrid_retriever( k=6 ),
        llm=llm,
        prompt=template,
        include_original=True
    )
    
    docs = mqr.invoke( state["question"] )

    doc_contents = []
    for d in docs:
        if hasattr( d, "page_content" ):
            doc_contents.append( d.page_content )

    reranker = __get_cross_encoder_reranker()

    # Cross-Encoder는 (question, doc)을 "같이" 넣고 점수를 예측함
    # 점수는 보통 "클수록 더 관련 있음"을 의미하며, 스케일은 모델마다 다름
    pairs = [(state["question"], doc) for doc in doc_contents]
    scores = reranker.predict( pairs )

    # scores는 모델 출력이라 numpy 배열/torch 텐서 같은 형태로 올 수 있습니다.
    # 아래에서 float(...)로 한 번 정규화해두면, 정렬/출력/로그가 다루기 쉬워집니다.
    scores_as_float = [float(s) for s in scores]

    # zip은 두개 리스트가 가진 원소를 하나씩 묶어 튜플로 만들어줌.
    # 다만, zip으로 묶을 준비만 하고, 실제 리스트로 만들려면 list로 변환 해야 함.
    # 결과적으로 튜플(score, doc) 리스트가 생김.
    score_doc = list( zip(scores_as_float, doc_contents) )
    
    # 리스트가 가진 튜플의 첫번째 값을 키로 사용 하여 정렬.
    # 파이썬에서는 튜플의 값을 배열의 인덱스처럼 사용하여 표현할 수 있음
    score_doc.sort( key=lambda x: x[0], reverse=True )

    # 점수 순으로 정렬된 리스트에서 첫 인덱스부터 최대 5개를 가져옴.
    top_scored = score_doc[: max(1, 5)]

    top_text = [text for (_, text) in top_scored]

    return { "documents" : top_text }

def node_evaluate(state:State):
    """
    LLM의 답변을 평가함. (RAGAS)
    라이브에서 쓰지 말 것.
    """

    # 평가 데이터
    q = state["question"]
    a = state["messages"][-1].content
    c = state.get( "documents", [] )
    r = state.get( "reference", "" )

    if not c or not a:
        return { "RAGAS 불가능. 질문이 없거나 질문에 사용된 문서가 없음" }
    
    # RAGAS 포맷에 맞게 데이터 구성
    data = {
        "user_input" : [q],
        "response": [a],
        "retrieved_contexts": [c],
        "reference": [r]
    }
    dataset = Dataset.from_dict( data )


    # 평가 시작
    print("\n[RAGAS 평가 진행 중...]")
    metrics = [
        faithfulness, # 충실도
        answer_relevancy, # 답변 관련성
        context_precision
    ]

    # evaluate 함수는 내부적으로 LLM(기본값 OpenAI)을 사용하여 점수를 매깁니다.
    # 커스텀 LLM을 쓰고 싶다면 metrics 초기화 시 llm을 바인딩해야 합니다.
    result = evaluate( 
        dataset, 
        metrics=metrics,
        raise_exceptions=False # 에러가 났을떄 그래프 전체가 죽는걸 방지함
    )


    # 결과 파싱
    score_df = result.to_pandas()
    score_dict = score_df.to_dict( "records" )[0]

    print(f"\n[RAGAS 평가 결과] {score_dict}")

    return { "ragas_score": score_dict }


# endregion

# state = State( {
#     "question": "1+1은 뭔가요?",
#     "messages": [ AIMessage( content="1+1은 2입니다." ) ],
#     "documents": [ "1. 덧셈은 두 수를 더하는 연산입니다.", "2. 1과 1을 더하면 2가 됩니다." ],
#     "reference": "2"
# } )
# result = node_evaluate( state )