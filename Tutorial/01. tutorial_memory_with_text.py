import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 1. 초기 설정
os.environ["OPENAI_API_KEY"] = "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" 
llm = ChatOpenAI(model='gpt-4o')
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# ========================================================
# 2. 데이터 적재 (기존과 동일)
# ========================================================
if not os.path.exists('./game_design.md'):
    with open('./game_design.md', 'w', encoding='utf-8') as f:
        f.write("# 게임 기획서\n## NPC\nNPC는 체력이 0이 되면 사망(Die) 상태가 된다.\n## 플레이어\n플레이어의 기본 공격력(ATK)은 10이다.")

loader = TextLoader('./game_design.md', encoding='utf-8')
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n", " "]
)
split_docs = text_splitter.split_documents(raw_documents)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    collection_name="game-design-standard-v1"
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# ========================================================
# 3. 체인 구성 (질문 변환 파트 - 기존과 동일)
# ========================================================

# 질의 변환용 템플릿
transform_prompt = ChatPromptTemplate.from_template("""
사전: {dictionary}
질문: {question}
수정된 질문(질문만 출력):
""")

question_transform_chain = { 
    "question": RunnablePassthrough(), 
    "dictionary": (lambda _: ["주인공 -> 플레이어", "피 -> 체력", "달면 -> 떨어지면" ])
} | transform_prompt | llm | StrOutputParser()


# ========================================================
# [중요] 대화 기억(Memory)을 위한 수동 관리 로직 추가
# ========================================================

# 1. 대화 내용을 저장할 리스트 (전역 변수처럼 사용)
# 형식: [("user", "질문내용"), ("ai", "답변내용"), ...]
chat_history = []

# 2. 대화 기록을 문자열로 예쁘게 포맷팅해주는 함수
# 체인 안에서 RunnableLambda로 실행될 것입니다.
def format_history(history_list):
    if not history_list:
        return "대화 기록 없음."
    
    formatted_text = ""
    for role, message in history_list:
        prefix = "User" if role == "user" else "AI"
        formatted_text += f"{prefix}: {message}\n"
    return formatted_text

# ========================================================
# 3-2. RAG 체인 구성 (수정됨)
# ========================================================

def format_docs(docs) -> str:
    texts = []
    for doc in docs:
        texts.append(doc.page_content)
    return "\n\n".join(texts)

# [수정 포인트 1] 프롬프트에 {history} 자리를 마련했습니다.
qa_prompt = ChatPromptTemplate.from_template("""
다음은 게임의 설정 자료(문맥)와 지금까지의 대화 내용입니다.
이를 바탕으로 질문에 답변하세요.

문맥(Context):
{context}

대화 기록(History):
{history}

질문: {input}
답변:
""")

# [수정 포인트 2] 체인 입력 부분에 'history' 키를 추가했습니다.
# 외부 변수 chat_history를 format_history 함수에 넣어 문자열로 변환한 뒤 주입합니다.
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough(),
        "history": RunnableLambda(lambda _: format_history(chat_history)) # 여기서 기억을 주입
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 최종 체인 (기존과 흐름 동일)
final_chain = (
    question_transform_chain | 
    { "input": RunnablePassthrough(), "answer": rag_chain}
)

# ========================================================
# 6. 실행 (시뮬레이션 루프)
# ========================================================
if __name__ == "__main__":
    print("--- Deep Space Terminal Game Start (type 'exit' to quit) ---")
    
    while True:
        # 1. 사용자 입력 받기
        query = input("\nUser: ")
        if query.lower() == "exit":
            break
            
        # 2. 체인 실행 (여기서 history는 위에서 정의한 람다를 통해 자동으로 주입됨)
        response = final_chain.invoke(query)
        answer_text = response['answer']
        
        # 3. 결과 출력
        print(f"AI: {answer_text}")
        
        # 4. [중요] 대화 내용 수동 업데이트
        # 이번 턴의 질문과 답변을 리스트에 추가합니다.
        # 다음번 invoke가 호출될 때, 이 내용이 format_history를 통해 프롬프트에 들어갑니다.
        # 변환된 질문(response['input'])을 저장할지, 원본 질문(query)을 저장할지는 선택입니다.
        # 여기서는 AI가 이해한 문맥을 유지하기 위해 변환된 질문을 저장하겠습니다.
        chat_history.append(("user", response['input'])) 
        chat_history.append(("ai", answer_text))