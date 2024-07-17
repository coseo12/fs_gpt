import streamlit as st
import openai
import glob
import math
import subprocess
from pydub import AudioSegment
import os

# 학습을 위한 불필요 동작 방지
has_transcript = os.path.exists("./.cache/podcast.txt")


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
