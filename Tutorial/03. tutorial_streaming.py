import os
from operator import itemgetter 

# 랭체인 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# [핵심] 대화 기억(Memory)을 위한 모듈
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 1. 초기 설정
# (기존에 입력하신 API 키를 그대로 사용하세요)
os.environ["OPENAI_API_KEY"] = "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" 

llm = ChatOpenAI(model='gpt-4o')
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# ========================================================
# 2. 데이터 적재 (Data Ingestion) - 기존과 동일
# ========================================================
if not os.path.exists('./game_design.md'):
    with open('./game_design.md', 'w', encoding='utf-8') as f:
        f.write("# 게임 기획서\n## NPC\nNPC는 체력이 0이 되면 사망(Die) 상태가 된다.\n## 플레이어\n플레이어의 기본 공격력(ATK)은 10이다.")

loader = TextLoader('./game_design.md', encoding='utf-8')
split_docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    collection_name="game-design-history-v1"
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# ========================================================
# 3. 세션(대화 기록) 저장소 정의
# ========================================================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ========================================================
# 4. 체인 구성 요소 준비
# ========================================================

# 4-1. 질문 변환
transform_prompt = ChatPromptTemplate.from_template("""
사전: {dictionary}
질문: {question}
수정된 질문(질문만 출력):
""")

question_transform_chain = (
    {
        "question": (lambda x: x["question"]), 
        "dictionary": (lambda _: ["주인공 -> 플레이어", "피 -> 체력", "달면 -> 떨어지면"])
    }
    | transform_prompt 
    | llm 
    | StrOutputParser()
)

# 4-2. RAG 응답 생성
qa_prompt = ChatPromptTemplate.from_template("""
다음은 게임의 설정 자료(문맥)와 지금까지의 대화 내용입니다.
이를 바탕으로 질문에 답변하세요.

문맥(Context):
{context}

대화 기록(History):
{history}

질문: {refined_question}
답변:
""")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": (lambda x: x["refined_question"]) | retriever | format_docs,
        "refined_question": (lambda x: x["refined_question"]),
        "history": (lambda x: x["history"]), 
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ========================================================
# 5. 전체 파이프라인 조립
# ========================================================
chain_with_logic = (
    RunnablePassthrough.assign(refined_question=question_transform_chain)
    | rag_chain
)

# ========================================================
# 6. RunnableWithMessageHistory로 감싸기
# ========================================================
final_chain = RunnableWithMessageHistory(
    runnable=chain_with_logic,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ========================================================
# 7. 실행 (Game Loop) - [변경됨] 스트리밍 적용
# ========================================================
if __name__ == "__main__":
    print("--- Deep Space Terminal (Memory V2 + Streaming) ---")
    session_id = "user_save_01"

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        print("AI: ", end="", flush=True) # 답변 시작 전 접두어 출력

        # [변경 핵심] invoke -> stream
        chunks = final_chain.stream(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        # [변경 핵심] 받아온 조각(chunk)을 실시간으로 출력
        for chunk in chunks:
            # StrOutputParser가 마지막에 있으므로 chunk는 단순 문자열(str)입니다.
            print(chunk, end="", flush=True)
        
        print() # 답변 완료 후 줄바꿈