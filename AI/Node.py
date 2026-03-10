import os
import sys
import json
from pathlib import Path
import ast

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
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import RemoveMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from ragas import evaluate
from ragas.metrics import Faithfulness, answer_relevancy, context_precision, ContextUtilization
from datasets import Dataset
from pprint import pprint
import dspy

import Utils.Utils as Utils
import config as Config
from AI.tools import all_tools

dotenv.load_dotenv()

PROJECT_ROOT = Utils.find_project_root( __file__ )

# 도구까지 바인딩 시켜놓음
llm = ChatOpenAI( model=Config.llm_model_name, temperature=0 )
llm_with_tool = llm.bind_tools( all_tools )

# DSPy 설정
api_key = os.getenv( "OPENAI_API_KEY", "" )
dspy_lm = dspy.LM( model=f"openai/{Config.llm_model_name}", api_key=api_key, temperature=0.0 )
dspy.configure( lm=dspy_lm )


# region LangGraph State
class State(TypedDict):
    question: Annotated[str, "question"] # 원본 질문. 그래프 루프가 시작할때 입력됨.
    documents: Annotated[list, "documents"] # 멀티쿼리 리트리버로 가져온 문서들.
    doc_names: Annotated[list, "document names"] # 멀티쿼리 리트리버로 가져온 문서들의 이름(출처)
    multi_queries: NotRequired[Annotated[list, "multi_queries"]] # 멀티쿼리로 생성된 쿼리들.
    messages: Annotated[list, add_messages] # 대화 히스토리
    summary: Annotated[str, "summary"] # 대화 요약본. 응답 후 마지막에 갱신 됨.
    reference: NotRequired[Annotated[str, "reference answer"]] # 평가용 정답. 라이브에서는 안쓰임.
    last_tool: NotRequired[Annotated[str, "last tool name"]] # 마지막으로 실행된 도구 이름(라우팅용)
    ragas: NotRequired[Annotated[dict, "ragas_scores"]] # RAGAS 평가 결과(메트릭 점수)

    opt_k: NotRequired[Annotated[int, "number of retrieved documents"]] # 리트리버가 가져온 문서 수 (디버깅/평가용)
    opt_top_k: NotRequired[Annotated[int, "number of documents after reranking"]] # 리랭킹 후 최종 문서 수 (디버깅/평가용)
    opt_w: NotRequired[Annotated[float, "weight for hybrid retriever"]] # 하이브리드 리트리버에서 가중치 (디버깅/평가용)
# endregion


# region DSPy Signature & Train set
class MultiQuerySignature(dspy.Signature):
    """사용자 질문을 분석하여 검색 쿼리를 3~5개 생성합니다."""
    input: str = dspy.InputField(desc="사용자의 질문")
    queries: list[str] = dspy.OutputField(desc="생성된 검색 쿼리 리스트")

class SummarySignature(dspy.Signature):
    """이전 대화의 요약본과 현재 대화 메세지를 검토하여 새로운 대화 요약본으로 업데이트 합니다.
    도구를 사용한 경우 사용 이유와 결과도 요약에 포함해세요.
    요약본만 출력하세요. """
    summary: str = dspy.InputField( desc="이전 대화 요약본" )
    message: str = dspy.InputField( desc="현재 대화 메시지" )
    new_summary: str = dspy.OutputField( desc="업데이트된 대화 요약본" )

mq_classifier = dspy.Predict(MultiQuerySignature)
summary_classsifier = dspy.Predict(SummarySignature)


@lru_cache(maxsize=1)
def __get_bm25_retriever(k: int) -> BM25Retriever:
    """
    BM25 리트리버 리턴.

    Args:
        k (int): 리트리버가 반환할 문서 수

    Returns:
        BM25Retriever: BM25 리트리버 객체
    """

    spliter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200 )
    chunks: list[Document] = spliter.split_documents( Utils.get_documents() )
    for c in chunks:
        c.metadata[ "category" ] = "docs"

    return BM25Retriever.from_documents( chunks, k=k )

@lru_cache(maxsize=1)
def __get_vector_retriever(k: int):
    """
    벡터 리트리버 리턴.

    Args:
        k (int): 리트리버가 반환할 문서 수

    Returns:
        BaseRetriever: 벡터 리트리버 객체
    """

    db = Utils.load_vector_db()
    return db.as_retriever( k=k )


#endregion

# region 유틸리티
@lru_cache(maxsize=1)
def __get_hybrid_retriever(k=6, w=0.5) -> EnsembleRetriever:
    """
    BM25 + Vector 결과를 rank fusion(Weighted RRF)으로 결합한 하이브리드 리트리버를 반환하는 유틸 함수.

    [BM25 리트리버]
    - 키워드 기반 전통적인 검색 방식 (TF-IDF 계열)
    - "정확히 이 단어가 문서에 얼마나 등장하는가"를 기준으로 점수화
    - 정확한 용어(고유명사, 날짜, 코드 등)가 포함된 질문에 강함
    - 의미적으로 비슷한 표현은 잡지 못함 (예: "유출" vs "노출")

    [Vector 리트리버]  
    - 임베딩(숫자 벡터) 기반 의미론적 검색 방식
    - 질문과 문서를 각각 고차원 벡터로 변환한 뒤, 벡터 간 거리(유사도)로 검색
    - 같은 의미의 다른 표현도 잡아냄 (예: "유출" ≈ "노출" ≈ "개인정보 침해")
    - 정확한 키워드가 없어도 맥락상 관련 문서를 가져올 수 있음

    [하이브리드 = BM25 + Vector]
    - 두 방식의 결과를 RRF(Reciprocal Rank Fusion)로 합산
    - 키워드 정밀도 + 의미 유연성을 동시에 확보
    - weights=[0.5, 0.5]로 두 리트리버의 영향력을 동등하게 설정

    Args:
        k (int): 각 리트리버에서 반환할 문서 수

    Returns:
        EnsembleRetriever: 하이브리드 리트리버 객체
    """

    # BM25 리트리버 준비
    bm25_retriever = __get_bm25_retriever( k )

    # Vector 리트리버 준비
    vector_retriever = __get_vector_retriever( k )

    
    # 하이브리드(앙상블) 리트리버
    return EnsembleRetriever(
        retrievers = [ bm25_retriever, vector_retriever ],
        weights = [ w, 1.0 - w ],
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
node_tools = ToolNode( all_tools, handle_tool_errors=True )



def node_input_question(state: State):
    """
    사용자의 질문을 입력받는 노드.
    현재 대화를 시작할때의 원본 질문을 저장해둔다.

    Returns:
        "question": 사용자의 질문
    """

    print( f"질문: {state['messages'][-1].content}" ) 

    return { "question": state["messages"][-1].content }



def node_tool_call(state: State):
    """
    질문을 보고 도구 사용이 필요할 경우 도구 사용.

    LLM의 출력은 크게 두 가지 중 하나인데, 사용자가 제어하는게 아님.
    1) 도구가 필요 없으면: 그냥 자연어 답변(content)을 가진 AIMessage
    2) 도구가 필요하면: tool_calls가 들어있는 AIMessage

    Returns:
        "messages": LLM의 답변 메시지
    """

    prompt = f"""
        당신은 쿠팡 사태의 모든것을 알고 있는 전문가입니다.
        주어진 질문과 정보를 바탕으로, 적합한 도구를 선택하여 사용하세요.

        [이전 대화 요약]
        {state.get("summary", "없음")}

        [질문]
        {state["question"]}

        [정보] 
        {state.get( "documents", "검색된 문서 없음" )}
    """

    #pprint( prompt, indent=2, width=100 )

    # 대화 내역은 프롬프트에 포함시키지 않고 따로 붙힌다.
    # state["messages"]에는 AIMessage,HumanMessage, ToolMessage와 같은 객체들을 리스트로 들고 있는데,
    # fstring으로 붙히게 되면 tostring 으로 처리 한 다음 치환 되기 때문에
    # 결국 리스트 자체가 tostring이 되어 버려 의미를 알 수 없게 되어 버림
    # 리스트의 tostring 모양 자체는 [AIMessage(content="..."), HumanMessage(content="...")] 이런식으로 되어서 문제가 없는 것 처럼 보일 수 있음
    # 따라서, 시스템 프롬프트를 먼저 만들고, 이후 대화 내역을 뒤로 붙혀서 리스트를 tostring 처리 없이 온전히 붙히는게 좋음.
    msg = [SystemMessage(content=prompt)] + state["messages"]

    decision = llm_with_tool.invoke( msg )

    # 사용할 도구가 없으면 그냥 넘김.
    # - tool_calls가 있으면 ToolNode가 실행할 수 있도록 messages에 decision(AIMessage)을 추가
    # - tool_calls가 없으면 여기서는 메시지를 추가하지 않고 final_answer 단계로 넘김
    if getattr(decision, "tool_calls", None):
        return { "messages": [decision] }

    return {}



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


def node_final_answer(state: State):
    """
    최종 답변 노드.
    현재 값들을 토대로 최종 답변 메세지를 생성한다.
    """

    prompt = f"""
        당신은 쿠팡 사태의 모든것을 알고 있는 전문가입니다.
        주어진 질문과 정보를 바탕으로, 최종 답변 메시지를 생성하세요.

        [이전 대화 요약]
        {state.get("summary", "없음")}

        [질문]
        {state["question"]}

        [정보] 
        {state.get( "documents", "검색된 문서 없음" )}
    """

    #print( prompt )

    msg = [SystemMessage(content=prompt)] + state["messages"]

    # answer = llm.invoke( msg )
    # return { "messages" : [answer] }

    # 최종 답변은 스트리밍으로 생성(청크를 소비하며 content를 누적)
    full_content = ""
    try:
        for chunk in llm.stream(msg):
            if isinstance(chunk, str):
                full_content += chunk
            else:
                full_content += getattr(chunk, "content", "") or ""
    except Exception:
        # 스트리밍 실패 시(모델/환경 이슈 등) 안전하게 단발 호출로 폴백
        answer = llm.invoke(msg)
        return {"messages": [answer]}

    answer = AIMessage(content=full_content)
    
    return {"messages": [answer]}




def node_multiquery(state:State):
    """
    사용자의 질문을 멀티쿼리리로 변환.

    Returns:
        "multi_queries": 멀티쿼리로 생성된 질문 리스트
    """

    # region 수동 프롬프트 방식
    # prompt = f"""
    #     사용자의 질문을 분석하여, 지식 베이스(매뉴얼)에서 정보를 가장 잘 찾을 수 있도록 검색 엔진용 쿼리를 5개 생성하세요. 
    #     각 쿼리는 서로 다른 키워드와 관점을 포함해야 합니다.
    #     각 쿼리는 한줄씩, 쿼리만 출력하세요.
                                 
    #     질문: {state["question"]}
    # """

    # mq_llm = ChatOpenAI( model=Config.query_llm_model_name, temperature=0 )
    # answer = mq_llm.invoke( prompt )
    # answer = getattr(answer, "content", "") or str(answer)
    # answers = answer.splitlines()
    
    # return {
    #     "multi_queries": answers
    # }
    #endregion 

    # DSPy로 프롬프트 입력 자동화.
    prediction: dspy.Prediction = mq_classifier.forward( input=state["question"] )

    return {
        "multi_queries": prediction.queries
    }
    

def node_hybrid_search(state:State):
    """
    앙상블 리트리버를 이용해 하이브리드 서치 결과를 가져온 다음
    결과를 토대로 리랭킹 까지 진행된 결과를 리턴함

    Returns:
        "documents" 에 검색/리랭크된 문서 텍스트 리스트를 넣음
        "doc_names" 에 검색/리랭크된 문서 이름 리스트를 넣음
    """

    # 최적화중에는 최적화된 값을 사용함.
    # 라이브중에는 상수처럼 사용
    k = state.get( "opt_k", Config.retriever_k )
    w = state.get( "opt_w", Config.retriever_w )

    queries = state.get("multi_queries", [])
    retriever = __get_hybrid_retriever( k, w )

    prompt = f"""
        당신은 쿠팡 사태의 모든것을 알고 있는 전문가입니다.
        주어진 [질문들]을 토대로 지식 베이스에서 정보를 검색하세요.

        [질문들]
        {queries}
    """

    docs = retriever.invoke( prompt )
    

    # Cross-Encoder는 (question, doc)을 "같이" 넣고 점수를 예측함
    # 점수는 보통 "클수록 더 관련 있음"을 의미하며, 스케일은 모델마다 다름
    #pairs = [(state["question"], doc) for doc in doc_contents]
    pairs: list[tuple[str, str]] = []
    doc_contents: list[str] = []
    for d in docs:
        if hasattr( d, "page_content" ):
            pairs.append( (state["question"], d.page_content) )

        if hasattr( d, "page_content" ):
            doc_contents.append( d.page_content )
        
    
    # 점수 계산
    reranker = __get_cross_encoder_reranker()
    scores = reranker.predict( pairs )
    
    # scores는 모델 출력이라 numpy 배열/torch 텐서 같은 형태로 올 수 있습니다.
    # 아래에서 float(...)로 한 번 정규화해두면, 정렬/출력/로그가 다루기 쉬워집니다.
    scores_as_float = [float(s) for s in scores]

    # 문서 제목
    doc_names: list[str] = []
    for d in docs:
        if hasattr( d, "page_content" ):
            doc_names.append( d.metadata.get( "source", "출처 불명" ) )

    # zip은 두개 리스트가 가진 원소를 하나씩 묶어 튜플로 만들어줌.
    # 다만, zip으로 묶을 준비만 하고, 실제 리스트로 만들려면 list로 변환 해야 함.
    # 결과적으로 튜플(score, doc) 리스트가 생김.
    score_doc = list( zip(scores_as_float, doc_contents, doc_names) )
    
    # 리스트가 가진 튜플의 첫번째 값을 키로 사용 하여 정렬.
    # 파이썬에서는 튜플의 값을 배열의 인덱스처럼 사용하여 표현할 수 있음
    score_doc.sort( key=lambda x: x[0], reverse=True )

    # 최적화중에는 최적화된 값을 사용함.
    # 라이브중에는 상수처럼 사용
    k = state.get( "opt_top_k", Config.reranking_top_k )

    # 점수 순으로 정렬된 리스트에서 첫 인덱스부터 최대 5개를 가져옴.
    top_scored = score_doc[: max(1, k) ]

    from pathlib import Path

    top_text = [text for (_, text, _) in top_scored]
    top_doc_name = [name for (_, _, name) in top_scored]
    top_doc_name = list(set(top_doc_name)) # 중복 제거
    doc_names: list[str] = []
    for dn in top_doc_name:
        doc_names.append( Path( dn ).name )
    

    return { "documents" : top_text, "doc_names" : doc_names }


def node_multiquery_search(state:State):
    """
    사용자의 질문을 여러개의 질문으로 변환(Multi-Query)하여, 
    앙상블 리트리버(Hybrid Search)를 이용해 RAG 검색 결과를 가져옴
    이 노드는 MultiQueryRetriever를 사용하여 멀티쿼리와 하이브리드 서치를 한방에 함.

    Returns:
        "documents" 에 검색/리랭크된 문서 텍스트 리스트를 넣음
    """
    
    # 하이브리드 서치로 잘 가져올 수 있도록,
    # 질문을 잘 나눌수 있게 멀티쿼리를 위한 프롬프트를 작성 해야 함.
    template = PromptTemplate.from_template( """
        사용자의 질문을 분석하여, 지식 베이스(매뉴얼)에서 정보를 가장 잘 찾을 수 있도록 검색 엔진용 쿼리를 5개 생성하세요. 
        각 쿼리는 서로 다른 키워드와 관점을 포함해야 합니다.
        각 쿼리는 한줄씩, 쿼리만 출력하세요.
                                 
        질문: {question}
    """ )

    # MultiQueryReriever는 멀티 쿼리와 리트리버를 이용한 검색을 한큐에 해버린다.
    # 이때 리트리트가 앙상블 리트리버라면 하이브리드 개념상 하이브리드 서치이므로,
    # Multi Query + Hybrid Search가 한큐에 되는 셈.
    # 다만, 멀티 쿼리로 질문이 5개가 된 경우, k가 6개라면, 질문당 문서를 6개를 가져오게 되는데,
    # 5 * 6 = 최대 30개의 문서가 검색될 수 있고, 이 문서를 전부 프롬프트로 날리면 토큰 사용량이 너무 커진다.
    # 그러므로 검색 후 적절히 걸러주는게 좋다. (Rerank 등..)
    # Unity로 전달할 메타데이터용: 멀티쿼리 생성 결과를 state에 남김
    try:
        mq_llm = ChatOpenAI(model=Config.query_llm_model_name, temperature=0)
        mq_answer = mq_llm.invoke(template.format(question=state["question"]))
        mq_text = getattr(mq_answer, "content", "") or str(mq_answer)
        multi_queries = [q.strip() for q in mq_text.splitlines() if q.strip()]
    except Exception:
        multi_queries = []

    k = state.get( "opt_k", Config.retriever_k )
    w = state.get( "opt_w", Config.retriever_w )

    mqr = MultiQueryRetriever.from_llm(
        retriever=__get_hybrid_retriever( k, w ),
        llm=llm,
        prompt=template,
        include_original=True
    )
    
    docs = mqr.invoke( state["question"] )
    
    pairs: list[tuple[str, str]] = []
    doc_contents: list[str] = []
    doc_names: list[str] = []
    for d in docs:
        if hasattr( d, "page_content" ):
            pairs.append( (state["question"], d.page_content) )
            doc_contents.append( d.page_content )
            doc_names.append( d.metadata.get( "source", "출처 불명" ) )

    reranker = __get_cross_encoder_reranker()

    # Cross-Encoder는 (question, doc)을 "같이" 넣고 점수를 예측함
    # 점수는 보통 "클수록 더 관련 있음"을 의미하며, 스케일은 모델마다 다름
    scores = reranker.predict( pairs )

    # scores는 모델 출력이라 numpy 배열/torch 텐서 같은 형태로 올 수 있습니다.
    # 아래에서 float(...)로 한 번 정규화해두면, 정렬/출력/로그가 다루기 쉬워집니다.
    scores_as_float = [float(s) for s in scores]

    # zip은 두개 리스트가 가진 원소를 하나씩 묶어 튜플로 만들어줌.
    # 다만, zip으로 묶을 준비만 하고, 실제 리스트로 만들려면 list로 변환 해야 함.
    # 결과적으로 튜플(score, doc, name) 리스트가 생김.
    score_doc = list( zip(scores_as_float, doc_contents, doc_names) )
    
    # 리스트가 가진 튜플의 첫번째 값을 키로 사용 하여 정렬.
    # 파이썬에서는 튜플의 값을 배열의 인덱스처럼 사용하여 표현할 수 있음
    score_doc.sort( key=lambda x: x[0], reverse=True )

    top_k = state.get( "opt_top_k", Config.reranking_top_k )

    # 점수 순으로 정렬된 리스트에서 첫 인덱스부터 최대 5개를 가져옴.
    top_scored = score_doc[: max(1, top_k)]

    top_text = [text for (_, text, _) in top_scored]
    top_doc_name = [name for (_, _, name) in top_scored]
    top_doc_name = list(set(top_doc_name)) # 중복 제거

    return {
        "documents": top_text,
        "doc_names": top_doc_name,
        "multi_queries": multi_queries,
    }




def node_summary_conversation(state: State):
    """
    대화 내역 요약.
    최종 답을 내리기 전 현재까지의 대화 내역을 요약한다.
    이 노드에 도달하면 messages에 있는 대화 내역을 토대로 summary를 업데이트 한다.

    Returns:
        "summary": 요약된 대화 내용
    """

    summary = state.get( "summary", "이전 대화 요약본 없음 (첫 대화일 경우 없을 수 있음)" )
    all_messages: list[BaseMessage] = state.get( "messages", [] )
    messages = __format_messages_for_summary( all_messages )

    # region 수동 프롬프트 방식
    # prompt = f"""
    #     아래는 시스템 엔지니어(사용자)와 AI 어시스턴트(당신) 간의 최신 [대화 내역]과 이전 대화 내역의 [요약본] 입니다.
    #     [대화 내역]과 [요약본] 을 보고 하나의 대화로 요약 해주세요.
    #     출력시 대화 요약된 대화 내역만 출력 하세요.
    #     현재 대화 세션에서 사용한 도구가 있다면 도구를 사용한 이유와, 그 결과도 요약 내용에 포함 하세요.
        
    #     [요약본]
    #     {summary}

    #     [대화 내역]
    #     {messages}
    # """

    # # 대화 요약 생성.
    # answer = llm.invoke( prompt )

    # print( f"대화 요약: {getattr(answer, 'content', str(answer))}" )

    # return { 
    #     "summary": getattr( answer, "content", str(answer) ) 
    # }
    # endregion

    prediction: dspy.Prediction = summary_classsifier.forward( summary=summary, message=messages )

    return {
        "summary": prediction.new_summary
    }




def node_evaluate(state:State):
    """
    LLM의 답변을 평가함. (RAGAS)
    라이브에서 쓰지 말 것.
    """

    # 평가 데이터
    # (주의) 사용자가 요청한 대로 "최종 답변 메시지 추출" 로직(#2)은 여기서 개선하지 않습니다.
    question = state.get("question", "")
    answer = state["messages"][-1].content if state.get("messages") else ""
    contexts = state.get("documents", [])
    reference = state.get("reference", "")

    # State를 더럽히지 않기 위해, 평가 결과는 state에 저장하지 않고 출력만 합니다.
    # (그래프 노드 반환값은 빈 dict로 처리)
    if not question or not contexts or not answer:
        print("[RAGAS] 스킵: question/answer/documents 중 하나가 비어있음")
        return {}

    # contexts는 list[str] 형태가 안전합니다.
    # (Document 객체나 None 등이 섞이면 ragas 내부에서 깨질 수 있으므로 문자열로 정규화)
    contexts = [str(c) for c in contexts if c]
    if not contexts:
        print("[RAGAS] 스킵: retrieved_contexts가 비어있음")
        return {}
    
    # RAGAS 포맷에 맞게 데이터 구성
    # - reference(정답)가 있는 경우에만 reference 기반 메트릭(context_precision)을 포함하는 쪽이 안전
    data = {
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
    }

    # 답변이 있다면 data에 reference(정답)도 포함.
    # 문맥 정밀도(context_precision), 문맥 재현율(context_recall)은 정답이 반드시 필요 하지만
    # 충실도(faithfulness), 답변 관련성(answer_relevancy)은 reference가 없어도 됨.
    # 그러므로, 정답이 입력이 없을 수 있음.
    # 문맥 활용도(context_utilization)도 정답이 필요하지 않음.
    has_reference = bool(str(reference).strip())
    if has_reference:
        data["reference"] = [str(reference)]

    # RAGAS 개발진은 HuggingFace의 datasets 라이브러리 구조를 채점용 입력 포맷으로 강제해뒀음.
    dataset = Dataset.from_dict( data )

    # 평가 시작
    # metrics는 평가 항목 리스트를 나타내며, ragas에서 미리 정의된 개체들로 사용함.
    print("\n[RAGAS 평가 진행 중...]")
    # ragas 0.4.x의 AnswerRelevancy는 기본 strictness=3으로 LLM에 n=3 generations을 요구합니다.
    # 일부 모델/환경에서는 n>1이 무시되어 내부에서 IndexError가 발생할 수 있어 strictness=1로 낮춥니다.
    from ragas.metrics import AnswerRelevancy

    # 각 메트릭의 instruction을 완전히 한국어로 교체.
    # (영어 원문 + 한국어 추가 방식은 LLM이 판단 기준을 혼동할 수 있으므로 한국어로 덮어씀)
    faithfulness_metric = Faithfulness()
    faithfulness_metric.statement_generator_prompt.instruction = (
        "질문과 답변이 주어지면, 답변의 각 문장을 분석하여 "
        "하나 이상의 완전히 이해 가능한 진술문으로 분해하세요. "
        "어떤 진술문에도 대명사를 사용하지 마세요. 결과를 JSON 형식으로 출력하세요."
    )
    faithfulness_metric.nli_statements_prompt.instruction = (
        "주어진 컨텍스트를 기반으로 각 진술문의 충실도를 판단하세요. "
        "컨텍스트에서 직접 추론 가능하면 verdict=1, 그렇지 않으면 verdict=0을 반환하세요."
    )

    # AnswerRelevancy는 가상 질문 생성 시 한국어로 하지 않으면
    # user_input(한국어)과의 임베딩 유사도가 0에 수렴하므로 반드시 한국어로 교체.
    # noncommittal: 답변이 애매하거나 모른다는 표현이면 1, 명확하면 0.
    answer_relevancy_metric = AnswerRelevancy(strictness=1)
    answer_relevancy_metric.question_generation.instruction = (
        "주어진 답변에 대한 질문을 한국어로 생성하세요. "
        "답변이 명확하고 구체적이면 noncommittal=0, "
        "모르겠다거나 애매하고 회피적인 표현이면 noncommittal=1로 표시하세요."
    )

    metrics = [
        faithfulness_metric,  # 충실도: 답변이 컨텍스트에 근거하는가?
        answer_relevancy_metric,  # 답변 관련성: 질문에 제대로 답했는가?
    ]
    
    # 그래서 답변 관련성이 0 뜨는 문제가 발생.
    # metrics = [
    #     fathfullness,  # 충실도: 답변이 컨텍스트에 근거하는가?
    #     AnswerRelevancy(strictness=1),  # 답변 관련성: 질문에 제대로 답했는가?
    # ]

    
    # reference(정답)가 있을 때만 context_precision을 포함
    # reference가 없으면 "정답 기반" 평가가 의미가 약해지므로 context_utilization(무참조)을 사용
    metrics.append(context_precision if has_reference else ContextUtilization())

    # 단순 버전: LangChain 기반 wrapper 사용
    # - ragas가 temperature를 0.01로 강제할 수 있어, bypass_temperature=True로 우회합니다.
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings

        embedding_model = Config.embedding_model_name
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # 평가용 LLM은 답변 생성용과 분리하는 게 안전합니다.
        # gpt-5/o-계열 등 temperature 제약 모델을 위해 temperature=1로 고정.
        eval_llm = ChatOpenAI(model=Config.llm_model_name, temperature=0)

        # NOTE: ragas 0.4.3에는 EvaluationResult 생성 시(트레이스 파싱) root_traces가 비어
        # IndexError가 발생하는 케이스가 있습니다. return_executor=True로 EvaluationResult
        # 생성을 우회하고, Executor.results()로 점수를 받아 직접 매핑합니다.
        executor = evaluate(
            dataset,  # 평가할 데이터
            metrics=metrics,  # 평가 항목
            llm=LangchainLLMWrapper(eval_llm, bypass_temperature=True),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
            raise_exceptions=False,
            return_executor=True,
        )
    except Exception as e:
        print(f"[RAGAS] 실패: {type(e).__name__}: {e}")
        return {}

    # 결과 파싱(현재 그래프는 dataset 1행 평가가 일반적)
    try:
        raw_scores = executor.results()
    except Exception as e:
        print(f"[RAGAS] 결과 수집 실패: {type(e).__name__}: {e}")
        return {}

    score_dict = {}
    if isinstance(raw_scores, list) and len(raw_scores) == len(metrics):
        normalized_scores = []
        for v in raw_scores:
            # numpy scalar -> python scalar
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            normalized_scores.append(v)

        score_dict = {getattr(m, "name", f"metric_{i}"): normalized_scores[i] for i, m in enumerate(metrics)}
    else:
        # 예외 케이스(예: 여러 행/내부 flatten 포맷)에서는 원시 결과를 그대로 노출
        score_dict = {"raw_scores": raw_scores}


    # 모든 메세지 삭제.
    # 다음 대화에서는 요약본을 토대로 대화하게 되어서 메세지는 필요 없음.
    messages: list[BaseMessage] = state.get( "messages", [] )
    remove_messages = []
    for m in messages:
        if m.id:
            remove_messages.append( RemoveMessage( m.id ) )


    print(f"\n[RAGAS 평가 결과] {score_dict}")

    return {
        "messages": remove_messages,
        "ragas": score_dict,
    }

def node_graph_end(state:State):
    """
    그래프 종료 노드.
    정리가 필요한 기능 등등을 여기에 작성 할 것.
    """

    print( "\n[그래프 종료]\n" )
    
    return {
        "multi_queries": state.get("multi_queries", []),
        "ragas": state.get("ragas", {}),
        "doc_names": state.get("doc_names", [])
    }

# endregion






"""이하 노드 테스트 코드들"""
# node_multiquery_search(State( {
#     "question": "얼마나 유출?",
# } ) )

# print( node_hybrid_search( State( {
#     "question" : "얼마나 유출?",
#     "multi_queries" : [ "얼마나 유출?", "몇명의 사용자의 정보가 유출 되었나요?"]
# } ) ) )