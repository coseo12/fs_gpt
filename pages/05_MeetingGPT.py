import streamlit as st
import openai
import glob
import math
import subprocess
import os
from pydub import AudioSegment
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# LLM 생성
llm = ChatOpenAI(
    temperature=0.1,
)

# 학습을 위한 불필요 동작 방지
has_transcript = os.path.exists("./.cache/podcast.txt")

# Define the text splitter
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


# 문서 형식
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 파일 처리
@st.cache_resource(
    show_spinner="Embedding file...",
)
def embed_file(file_path):
    # cache_dir - 캐시 디렉토리
    cache_dir = LocalFileStore(f"./.cache/meeting_embeddings/{file.name}")
    # chunk_size - 텍스트를 분할하는 크기
    # chunk_overlap - 분할된 텍스트의 중복 크기
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    # 캐시된 임베딩을 사용하여 Vector Store 초기화
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    # Vector Store 초기화
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver


# extract_audio_from_video 함수를 사용하여 오디오 추출
@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


# 오디오를 10분 단위로 자르기
@st.cache_data()
def cutting_audio_into_chunks(audio_path, chunks_path):
    if has_transcript:
        return
    track = AudioSegment.from_file(audio_path)
    ten_minutes = 10 * 60 * 1000
    chunks = math.ceil(len(track) / ten_minutes)
    for i in range(chunks):
        start_time = i * ten_minutes
        end_time = (i + 1) * ten_minutes
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_path}/chunk_{i}.mp3", format="mp3")


# Get Audio to Text
@st.cache_data()
def transcribe_chunks(chunk_path, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_path}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            # transcribe audio
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", file=open(audio_file.name, "rb"), language="en"
            )
            text_file.write(transcript.text)


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="📆",
)

st.title("MeetingGPT")

st.markdown(
    """
    Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

    Get started by uploading a video file in the sidebar.
    """
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
if video:
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        chunks_path = "./.cache/chunks"
        destination_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cutting_audio_into_chunks(audio_path, chunks_path)
        status.update(label="Trascribing audio...")
        transcribe_chunks(chunks_path, destination_path)

    trascript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with trascript_tab:
        with open(destination_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate Summary")

        if start:
            # Load the documents
            loader = TextLoader(destination_path)

            # Load and split the documents
            docs = loader.load_and_split(text_splitter=splitter)

            # Define the prompt for the first summary
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{context}"
                CONCISE SUMMARY:
                """
            )

            # Define the chain of operations
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            # Summarize the first document
            summary = first_summary_chain.invoke({"context": docs[0].page_content})

            # Define the prompt for refining the summary
            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We habe the opportunity to refine the existing summary (only if needed) with some more context below.
                ----------
                "{context}"
                ----------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original context.
                """
            )

            # Define the chain of operations
            refine_chain = refine_prompt | llm | StrOutputParser()

            # Summarize the rest of the documents
            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs) - 1}")
                    summary = refine_chain.invoke(
                        {"existing_summary": summary, "context": doc.page_content}
                    )
                    st.write(summary)
            st.write(summary)

    with qa_tab:
        retriever = embed_file(destination_path)

        qustion = st.text_input(label="Ask a question")

        qna_prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using Only the following context, If you don't know the answer 
            just say you don't know. DON'T make anything up.
            If you ask a question in not english, we will translate the context and process it.

            Context: {context}
            Question: {question}                                         
            """
        )

        if qustion:
            qna_chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | qna_prompt
                | llm
            )

            with st.chat_message("ai"):
                res = qna_chain.invoke(qustion)
                st.markdown(res.content)
