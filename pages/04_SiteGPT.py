import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


# Load the website
@st.cache_data(show_spinner="Loading...website")
def load_website(url):
    # splitter - 텍스트를 분할하는 방법
    # chunk_size - 텍스트를 분할하는 크기
    # chunk_overlap - 분할된 텍스트의 중복 크기
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # Load the website
    # filter_urls - 필터링할 URL
    # parsing_function - 파싱 함수
    loader = SitemapLoader(
        url,
        filter_urls=[
            # "https://openai.com/index/data-partnerships",
            r"^(.*\/index\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌏",
)

st.title("SiteGPT")


st.markdown(
    """
    Ask questions about the content of a website.
    
    Start by writing the URL of the website on the sidebar.
    """
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://www.example.com")


if url:
    # Check if the URL is a SiteMap
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a SiteMap URL")
    else:
        # Load the SiteMap
        docs = load_website(url)
        st.write(docs)
