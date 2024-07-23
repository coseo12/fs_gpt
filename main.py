from fastapi import FastAPI
from pydantic import BaseModel
from pinecone.grpc import PineconeGRPC as Pinecone
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Pinecone 초기화
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 임베딩 초기화
embeddings = OpenAIEmbeddings()

# Pinecone Vector Store 초기화
vectore_store = PineconeVectorStore.from_existing_index(
    "chefgpt",
    embeddings,
)

app = FastAPI(
    title="ChefGPT. The best probvider of Indian Recipes in the world.",
    description="Give ChefGPT a couple of ingredients and it will give you recipes in return.",
    servers=[{"url": "https://protocols-compaq-shanghai-vids.trycloudflare.com"}],
)


class Document(BaseModel):
    page_content: str


@app.get(
    "/recipe",
    summary="Returns a list of recipes.",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient.",
    response_description="A Document object that cntains the recipe and preparation instructions.",
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": True,
    },
)
def get_recipe(ingredient: str):
    # 유사도 검색
    docs = vectore_store.similarity_search(ingredient)
    return docs
