import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
from pathlib import Path
from dotenv import load_dotenv

# 1. OpenAI 모델 및 임베딩 (langchain-openai)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 2. 벡터 저장소 (langchain-chroma) 
# ★ 중요: requirements에 langchain-chroma를 설치했으므로 여기서 불러옵니다.
# 기존: from langchain_community.vectorstores import Chroma (더 이상 사용 X)
from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config as Config 

def load_vector_db(remake_db=False) -> Chroma:
    """
    벡터 DB를 불러오거나 새로 만듭니다.
    
    Args:
        remake_db: True로 설정 시 기존 DB가 있더라도 새로 만듦.
    """

    db_path = Config.vector_db_path
    collection_name = Config.collection_name
    embedding = OpenAIEmbeddings( model=Config.embedding_model_name )

    # 벡터DB를 만들거나 불러오기.
    # 응답시 어떤 여러 컬렉션의 정보가 필요할 경우 "멀티 리트리버", "라우터 체인" 같은 기술을 쓴다고 하지만
    # 실무에서는 컬렉션 하나에 몰빵하고 메타데이터로 구분한다고 함. (잼민이나 그랬음..)
    # 더 간단하고, 문서의 종류 추가시 컬렉션을 추가 하는등의 유지보수 없이 메타데이터만 추가하면 되기 때문.
    # 벡터DB는 인덱싱 기술 덕분에 성능문제가 별로 없다고 함
    if remake_db is True or not os.path.exists( db_path ):

        # 임베딩 한다 라는것: 벡터화 한다라는것.
        # 임베딩 하기 위한 사전 준비. 텍스트를 나누기 위한 splitter 생성.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        all_docs = []
        extensions = [ ".txt", ".pdf" ]
        for f in os.listdir( Config.doc_folder_path ):

            #if f.endswith( tuple( extensions ) ) == False:

            # 문서 로드
            #text_loader = TextLoader( f"{Config.doc_folder_path}/{f}", encoding="utf8" )
            text_loader = PyPDFLoader( f"{Config.doc_folder_path}/{f}" )

            # 청킹
            docs = text_splitter.split_documents( text_loader.load() )
            for d in docs:
                #d.metadata = { "category" : "rule" } # 불러온 메타데이터가 날아가버림.
                d.metadata[ "category" ] = "docs" # 문서의 성격별로 구분 하려면??

            all_docs.extend( docs )


        vector_db = None

        # 벡터 DB 생성
        vector_db = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=db_path,
        )

        print( "벡터 DB 생성" )

    # 기존에 만들어진 DB가 있다면 그대로 씀
    else:
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=collection_name,
            persist_directory=db_path,
        )

        print( "기존 벡터 DB 사용" )
    
    return vector_db

def get_documents() -> list[Document]:
    """
    문서 폴더 하위에 있는 모든 문서를 Document 리스트로 반환함
    현재는 pdf만 됨

    Returns:
        list[Document]: 문서 리스트
    """

    all_docs: list[Document] = []
    for f in  os.listdir( Config.doc_folder_path ):
        if f.endswith(".pdf") == False:
            continue
        loader = PyPDFLoader( f"{Config.doc_folder_path}/{f}" )
        docs = loader.load()
        all_docs.extend( docs )

    return all_docs
            

def find_project_root(file_path:str):
    # 1. 현재 이 스크립트 파일의 경로를 가져옵니다.
    current_file_path = Path(file_path).resolve()
    
    # 2. 파일 이름은 빼고, 파일이 속한 '폴더' 경로만 남깁니다.
    current_folder = current_file_path.parent
    
    # 3. '현재 폴더'부터 시작해서 '최상위 루트(/)'까지 하나씩 위로 올라가며 검사합니다.
    # [현재폴더, 부모폴더, 조부모폴더...] 순서입니다.
    for folder in [current_folder] + list(current_folder.parents):
        
        # 4. 해당 폴더 안에 '루트'임을 증명하는 파일이 있는지 체크합니다.
        git_folder = folder / ".git"
        req_file = folder / "requirements.txt"
        
        if git_folder.exists() or req_file.exists():
            # 찾았다면 그 폴더 경로를 반환합니다.
            return folder
            
    # 끝까지 못 찾으면 그냥 현재 폴더를 반환합니다.
    return current_folder

#PROJECT_ROOT = find_root(Path(__file__)