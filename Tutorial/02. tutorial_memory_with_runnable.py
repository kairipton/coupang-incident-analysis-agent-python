import os
from operator import itemgetter # 딕셔너리에서 값 꺼낼 때 쓰는 표준 도구

# 랭체인 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# [핵심] 대화 기억(Memory)을 위한 모듈
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 1. 초기 설정
# (실제 API 키로 교체 필요)
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
# 여기가 '세이브 파일'이 저장되는 메모리 공간입니다.
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """session_id에 해당하는 대화 기록을 반환하거나 새로 만듭니다."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ========================================================
# 4. 체인 구성 요소 준비
# ========================================================

# 4-1. 질문 변환 (단어 치환)
# itemgetter("question"): 입력된 딕셔너리에서 'question' 키의 값만 쏙 빼냅니다.
transform_prompt = ChatPromptTemplate.from_template("""
사전: {dictionary}
질문: {question}
수정된 질문(질문만 출력):
""")

question_transform_chain = (
    {
        # 입력(x)에서 'question' 키의 값을 꺼냄
        "question": (lambda x: x["question"]), 
        
        # 입력(x)는 안 쓰지만, 사전 리스트를 리턴함
        "dictionary": (lambda _: ["주인공 -> 플레이어", "피 -> 체력", "달면 -> 떨어지면"])
    }
    | transform_prompt 
    | llm 
    | StrOutputParser()
)


# 4-2. RAG 응답 생성
# 여기서 {history}가 사용됩니다.
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

# RAG 체인
# 입력으로 들어오는 딕셔너리에는 {refined_question, history, question}이 다 들어있습니다.
rag_chain = (
    {
        # 입력(x)에서 'refined_question'을 꺼내서 리트리버에 넣고 포맷팅
        "context": (lambda x: x["refined_question"]) | retriever | format_docs,
        
        # 입력(x)에서 'refined_question'을 그대로 패스
        "refined_question": (lambda x: x["refined_question"]),
        
        # 입력(x)에서 'history'를 꺼냄
        "history": (lambda x: x["history"]), 
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ========================================================
# 5. 전체 파이프라인 조립 (RunnablePassthrough.assign 사용)
# ========================================================

# RunnablePassthough.assign: 기존 입력값을 유지하고, 기존 입력값을 토대로 새로운 값을 추가함.
# 흐름상 이 체인이 호출될때는 메인루프에서 호출되서 가장 먼저 시작되는 체인인데,
# question, history 값 두개만 있으며, 이 값은 그대로 다음 체인으로 전달하게 된다.
# 그리고, refined_question이라는 이름으로 새로운 값을 전달 하는데, 이 값은 question_transform_chain의 결과값이 된다.
# 그러므로, rag_chain 순서가 될때는 question, history, refined_question 3개의 값을 전달하게 된다.
# refined_question라는 파라미터는 파이썬의 가변인자 문법이므로 사용자가 이름을 정할 수 있음.
chain_with_logic = (
    RunnablePassthrough.assign(refined_question=question_transform_chain)
    | rag_chain
)

# ========================================================
# 6. [핵심] RunnableWithMessageHistory로 감싸기
# ========================================================
# 기억력을 가진 AI로 만들려면 체인을 파이프로 단순히 엮는게 아니라 RunnableWithMessageHistory로 만들어야 함
# 메인루프에서 invoke를 호출 할때 dictioary로 question값을 주는데, input_messages_key에 해당 키값을 주면 되는 듯
# get_session_history에 넣은 메서드로 대화 내역을 가져오게 만들고, 대화 내역은 history라는 키로 주입됨.
final_chain = RunnableWithMessageHistory(
    runnable=chain_with_logic, # 실제 체인 로직.
    get_session_history=get_session_history, # 대화 내역을 가져오는 메서드
    input_messages_key="question", # 입력 딕셔너리의 어떤 키가 '사용자 질문'인가?
    history_messages_key="history", # 프롬프트나 체인 내부에서 '대화 기록'을 어떤 키로 받을 것인가?
)

# ========================================================
# 7. 실행 (Game Loop)
# ========================================================
if __name__ == "__main__":
    print("--- Deep Space Terminal (Memory V2) ---")
    session_id = "user_save_01" # 이 ID가 같으면 계속 기억합니다.

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # invoke 할 때 config에 session_id를 전달하는 것이 규칙입니다.
        # 사용자 입력값을 dictionary로 전달해야 한다. 기억력을 가진 AI(RunnableWithMessageHistory)를 작업하려면 이게 규칙인듯
        response = final_chain.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"AI: {response}")