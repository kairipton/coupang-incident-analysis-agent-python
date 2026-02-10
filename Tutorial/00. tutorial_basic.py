import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# [Standard] 데이터 흐름 제어를 위한 표준 도구
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 1. 초기 설정
os.environ["OPENAI_API_KEY"] = "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" 
llm = ChatOpenAI(model='gpt-4o')
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# ========================================================
# 2. 데이터 적재 (Data Ingestion)
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

# 벡터DB를 통해 바로 질문해도 되지만, 리트리버로 변환하는것은 랭체인에서 리트리버 라는 규격을 요구 하기 때문. 인터페이스를 생각하면 됨.
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})


# ========================================================
# 3. 체인 구성
# ========================================================



def inject_dictionary(_):
    return dictionary

# 질의 변환용 맵
dictionary = ["주인공 -> 플레이어", "피 -> 체력", "달면 -> 떨어지면" ]

# 질의 변환용 템플릿 구성.
transform_prompt = ChatPromptTemplate.from_template("""
사전: {dictionary}
질문: {question}
수정된 질문(질문만 출력):
""")

# ========================================================
# LCEL 파이프 구성
# Runnable 인터페이스를 사용하는 클래스를 체인으로 구성할 수 있고, Dictionary나 람다는 편의를 위해 Wrapper 클래스를 씌워서 Runnable로 변환 시켜줌.
# 체인에 구성된 Runnable은 체인을 타고 흐르면서 순서가 되면 자동으로 실행 됨.
# RunnablePassthrough: Invoke등으로 입력 받은 값을 그대로 다음으로 넘겨줌.
# ========================================================
# 질문 받은 내용(RunnablePassthrough)와 dictionary(RunnableLambda)를
# transform_prompt에 주입하고 | llm에 질의 한 다음 | 답변된 내용중 문자열만 추출해옴.
# 결과적으로 변환된 질문의 내용만 저확히 얻게 되는 체인.
question_transform_chain = { 
        #"question": RunnablePassthrough(), "dictionary": RunnableLambda(inject_dictionary) 

        # 이렇게 람다를 바로 써도 됨
        "question": RunnablePassthrough(), "dictionary": (lambda _: ["주인공 -> 플레이어", "피 -> 체력", "달면 -> 떨어지면" ])
    } | transform_prompt | llm | StrOutputParser()

# ========================================================
# 3-2. RAG 체인 구성
# ========================================================

def format_docs(docs) -> str:
    texts = []
    for doc in docs:
        texts.append(doc.page_content)  # 각 문서의 본문 텍스트만 모음

    result = "\n\n".join(texts)  # 빈 줄로 구분해서 하나로 합침
    return result

qa_prompt = ChatPromptTemplate.from_template("""
    문맥: {context}
    질문: {input}
    답변:
""")

# question_transform_chain에서 받는 문자열을 사용하게 됨.
# 리트리버는 Runnable이므로 이전 체인에서 받은 값으로 실행하여 검색된 결과를 가져오고, 그 다음 RunnableLambda로 문서 포맷팅을 수행.
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough(),
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)



# 최종 체인 구성.
# 질문을 수정하고 (question_transform_chain)
# 마지막 체인을 dictionary로 구성하는데, dictionary는 랭체인에서 Runnable로 wrapping하므로 Runnable로 취급 됨
# 결국 dictionary로 Runnable이므로 체인의 구성요소로 실행 되고, rag_chain도 Runnable이므로 체인의 구성요소로 실행 되어 answer를 얻게 됨.
final_chain = (
    question_transform_chain | 
    { "input": RunnablePassthrough(), "answer": rag_chain}
)

# ========================================================
# 6. 실행
# ========================================================
if __name__ == "__main__":
    query = "주인공 피 다 달면 어떻게 돼?"
    print(f"--- 질문: {query} ---")
    
    # 이제 invoke에는 'question' 하나만 넣으면 알아서 사전 주입되고 실행됩니다.
    response = final_chain.invoke(query)
    
    #print(f"수정된 질문: {response['input']}")
    #print(f"최종 답변: {response['answer']}")
    print( response )