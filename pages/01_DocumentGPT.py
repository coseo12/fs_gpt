import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import Chroma

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📜",
)


# 파일 처리
@st.cache_resource(
    show_spinner="Embedding file...",
)
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    # cache_dir - 캐시 디렉토리
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    # chunk_size - 텍스트를 분할하는 크기
    # chunk_overlap - 분할된 텍스트의 중복 크기
    # separator - 텍스트를 분할하는 구분자
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
        separator="\n",
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    # 캐시된 임베딩을 사용하여 Vector Store 초기화
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    # Vector Store 초기화
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver


# 메시지 전송
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


# 이전 메시지 표시
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

# 사이드바
with st.sidebar:
    # 파일 업로드
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
    )

if file:
    rr = embed_file(file)

    send_message("I'm ready! Ask away!", "ai", save=False)

    paint_history()

    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
else:
    st.session_state["messages"] = []
