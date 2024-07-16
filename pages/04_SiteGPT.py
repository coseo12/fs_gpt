import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Define the language model
llm = ChatOpenAI(
    temperature=0.1,
)

# Define the prompt
ansers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
    """
)


# Define the function to get the answers
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = ansers_prompt | llm
    answers = []
    for doc in docs:
        result = answers_chain.invoke(
            {
                "context": doc.page_content,
                "question": question,
            }
        )
        answers.append(result)
    st.write(answers)


# Define the function to parse the page
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


# Load the website
@st.cache_resource(show_spinner="Loading...website")
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
            # r"^(.*\/index\/).*",
        ],
        parsing_function=parse_page,
    )
    # requests_per_second - 초당 요청 수
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    # vector_store - 문서 벡터 저장소
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


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
        retriever = load_website(url)

        chain = {
            "docs": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answers)

        chain.invoke("Ways to partner with us?")
