import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌏",
)

st.title("SiteGPT")

# Transformer to convert HTML to text
html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    Ask questions about the content of a website.
    
    Start by writing the URL of the website on the sidebar.
    """
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://www.example.com")


if url:
    # Load the website
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    # Transform the HTML to text
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)
