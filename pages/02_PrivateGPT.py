import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


# 콜백 핸들러
class ChatCallbackHandler(BaseCallbackHandler):
    # 메시지
    message = ""

    # llm_start 이벤트
    def on_llm_start(self, *args, **kwargs):
        # 메시지 박스 ( 비어있는 위젯 )
        self.message_box = st.empty()

    # llm_end 이벤트
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    # llm_new_token 이벤트
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = Ollama(
    model="llama3:latest",  # 모델 선택
    temperature=0.1,  # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
    # streaming=True,  # streaming 옵션을 활성화하여 대화형 모드로 설정 (Ollama에서는 지원하지 않음)
    callbacks=[ChatCallbackHandler()],  # 콜백 함수를 설정
)

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="🔒",
)


# 파일 처리
@st.cache_resource(
    show_spinner="Embedding file...",
)
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    # cache_dir - 캐시 디렉토리
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
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
    embeddings = OllamaEmbeddings(model="llama3:latest")
    # 캐시된 임베딩을 사용하여 Vector Store 초기화
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    # Vector Store 초기화
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver


# 메시지 저장
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# 메시지 전송
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# 이전 메시지 표시
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


# 문서 형식
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 템플릿
prompt = ChatPromptTemplate.from_template(
    """Answer the question using Only the following context, If you don't know the answer 
    just say you don't know. DON'T make anything up.
    If you ask a question in not english, we will translate the context and process it.

    Context: {context}
    Question: {question}
    """
)

st.title("PrivateGPT")

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
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        # 대화 체인
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            # 메시지 전송
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
