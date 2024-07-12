import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🧐",
)


# 파일 처리
@st.cache_resource(
    show_spinner="Loading file...",
)
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

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
    return docs


st.title("QuizGPT")

with st.sidebar:
    choice = st.selectbox("Choose what you want to use", ("File", "Wikipedia Article"))

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file", type=["docx", "pdf", "txt"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            # top_k_results - Wikipedia에서 가져올 결과의 수
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching wikipedia..."):
                # Wikipedia에서 관련 문서를 가져옴
                docs = retriever.get_relevant_documents(topic)
