import sys
import os

# 프로젝트 루트를 sys.path에 추가 (AI, Utils 등 패키지를 import 하기 위함)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

import AI.Node as Node


# ─── 1. node_route_next ───────────────────────────────────────────────────────

def test_node_route_next_returns_need_tools_when_tool_calls_exist():
    """
    마지막 메시지에 tool_calls가 있으면 "need_tools"를 리턴하는지 확인.
    """
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {}, "id": "call_1", "type": "tool_call"}],
    )
    state = {"messages": [ai_msg]}

    result = Node.node_route_next(state)
    
    assert result == "need_tools"


def test_node_route_next_returns_NONE_when_no_tool_calls():
    """
    마지막 메시지에 tool_calls가 없으면 "NONE"을 리턴하는지 확인.
    """
    ai_msg = AIMessage(content="일반 답변입니다. 도구는 필요 없어요.")
    state = {"messages": [ai_msg]}

    result = Node.node_route_next(state)

    assert result == "NONE"


# ─── 2. node_hybrid_search ────────────────────────────────────────────────────

def test_node_hybrid_search_sorts_documents_by_reranker_score():
    """
    리랭킹 결과로 점수가 높은 문서가 앞에 오는지 확인.
    score_doc.sort(key=lambda x: x[0], reverse=True) 로직 검증.

    리트리버가 [낮은점수, 높은점수] 순으로 반환하더라도,
    리랭크 후에는 높은 점수 문서가 첫 번째에 위치해야 한다.
    이 테스트가 실패하면 정렬 방향(reverse=True)이 뒤집혔거나 로직이 바뀐 것.
    """
    # 리트리버가 낮은점수 문서를 먼저 반환하는 상황
    fake_docs = [
        Document(page_content="낮은 점수 문서", metadata={"source": "/path/low.txt"}),
        Document(page_content="높은 점수 문서", metadata={"source": "/path/high.txt"}),
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = fake_docs

    # 리랭커 점수: 낮은점수 문서=0.2, 높은점수 문서=0.9
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.2, 0.9]

    state = {
        "question": "쿠팡 개인정보 유출 규모는?",
        "multi_queries": [],
    }

    with patch("AI.Node.__get_hybrid_retriever", return_value=mock_retriever), \
         patch("AI.Node.__get_cross_encoder_reranker", return_value=mock_reranker):
        result = Node.node_hybrid_search(state)

    # 점수 높은 문서가 첫 번째에 와야 함
    assert result["documents"][0] == "높은 점수 문서"


# ─── 3. node_multiquery_search ────────────────────────────────────────────────

def test_node_multiquery_search_includes_multi_queries_in_result():
    """
    ChatOpenAI로 생성된 멀티 쿼리가 반환값의 multi_queries에 포함되는지 확인.
    multi_queries는 Unity 클라이언트에 진행 상태를 표시하는 메타데이터로 쓰인다.

    AIMessage의 content를 줄바꿈 기준으로 쪼갠 결과가 multi_queries에 담겨야 한다.
    """
    fake_docs = [
        Document(page_content="유출 문서 내용", metadata={"source": "leak.txt"}),
    ]
    mock_mqr = MagicMock()
    mock_mqr.invoke.return_value = fake_docs

    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.8]

    state = {"question": "쿠팡 사태란?"}

    with patch("AI.Node.ChatOpenAI") as mock_chat, \
         patch("AI.Node.MultiQueryRetriever") as mock_mqr_class, \
         patch("AI.Node.__get_hybrid_retriever"), \
         patch("AI.Node.__get_cross_encoder_reranker", return_value=mock_reranker):

        # mq_llm.invoke(...)가 줄바꿈 구분 쿼리 목록 반환
        mock_chat.return_value.invoke.return_value = MagicMock(content="쿠팡 유출\n개인정보 규모")
        mock_mqr_class.from_llm.return_value = mock_mqr

        result = Node.node_multiquery_search(state)

    assert "documents" in result
    assert "doc_names" in result
    assert "multi_queries" in result
    # content를 줄 단위로 쪼갠 결과
    assert result["multi_queries"] == ["쿠팡 유출", "개인정보 규모"]


# ─── 4. lru_cache ─────────────────────────────────────────────────────────────

def test_lru_cache_bm25_only_loads_documents_once():
    """
    __get_bm25_retriever(k)를 동일한 인자로 두 번 호출해도
    Utils.get_documents()는 단 1번만 실행되는지(캐시 동작) 확인.

    lru_cache는 같은 인자의 두 번째 호출부터 캐시된 결과를 바로 반환하므로
    함수 내부 로직(문서 로딩, BM25 인덱싱)이 다시 실행되지 않아야 한다.

    Node.__dict__['__get_bm25_retriever'] 로 함수 객체를 꺼내는 이유:
    - 모듈 레벨의 __name 은 파이썬 name-mangling 대상이 아님
    - 단, 클래스 바깥의 코드에서도 직접 접근하기가 어려우므로
      __dict__ 를 통해 명시적으로 꺼낸다
    """
    get_bm25 = Node.__dict__["__get_bm25_retriever"]

    # 이전 테스트나 import 시점에 캐시가 채워졌을 수 있으므로 초기화
    get_bm25.cache_clear()

    with patch("AI.Node.Utils") as mock_utils, \
         patch("AI.Node.BM25Retriever") as mock_bm25, \
         patch("AI.Node.RecursiveCharacterTextSplitter") as mock_splitter:

        mock_utils.get_documents.return_value = []
        mock_splitter.return_value.split_documents.return_value = []
        mock_bm25.from_documents.return_value = MagicMock()

        get_bm25(2)  # 첫 번째 호출: 실제로 내부 로직 실행 후 결과를 캐시에 저장
        get_bm25(2)  # 두 번째 호출: 캐시에서 바로 반환 (내부 로직 실행 안 함)

    # 내부 로직(문서 로딩)은 오직 1번만 실행됐어야 함
    assert mock_utils.get_documents.call_count == 1
