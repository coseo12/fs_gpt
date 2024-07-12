import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🧐",
)

llm = ChatOpenAI(
    temperature=0.1,  # temperature - 다양성을 조절하는 매개변수
    model="gpt-3.5-turbo-1106",  # model - 사용할 모델
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


# 문서 형식
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a helpfull assistant that is role playing as a teacher.
         
        Based ONLY on the following context make 10 questions to test the user's
        knowledge about the text.

        Each question should have 4 answers, three of them must be incorrect and
        one should be correct.

        Use (o) to signal the correct answer.

        Question example:
        
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
        
        Question: What is the capital or Georgia?
        Answers: Bakum Tbilisi(o)|Manila|Beirut

        Question: When was Avartar released?
        Answers: 2007|2001|2009(o)|1998

        Question: Who was Julius Caesar?
        Answers: A Roamn Emperor(o)|Painter|Actor|Model

        Your turn!

        Context: {context}
        """,
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.
     
            You format exam questions into JSON format.
            Answers with (o) are the correct ones.
            
            Example Input:
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
                
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
                
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
                
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
            
            
            Example Output:
            
            ```json
            {{ "questions": [
                    {{
                        "question": "What is the color of the ocean?",
                        "answers": [
                                {{
                                    "answer": "Red",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Yellow",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Green",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Blue",
                                    "correct": true
                                }},
                        ]
                    }},
                                {{
                        "question": "What is the capital or Georgia?",
                        "answers": [
                                {{
                                    "answer": "Baku",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Tbilisi",
                                    "correct": true
                                }},
                                {{
                                    "answer": "Manila",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Beirut",
                                    "correct": false
                                }},
                        ]
                    }},
                                {{
                        "question": "When was Avatar released?",
                        "answers": [
                                {{
                                    "answer": "2007",
                                    "correct": false
                                }},
                                {{
                                    "answer": "2001",
                                    "correct": false
                                }},
                                {{
                                    "answer": "2009",
                                    "correct": true
                                }},
                                {{
                                    "answer": "1998",
                                    "correct": false
                                }},
                        ]
                    }},
                    {{
                        "question": "Who was Julius Caesar?",
                        "answers": [
                                {{
                                    "answer": "A Roman Emperor",
                                    "correct": true
                                }},
                                {{
                                    "answer": "Painter",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Actor",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Model",
                                    "correct": false
                                }},
                        ]
                    }}
                ]
            }}
            ```
            Your turn!
            Questions: {context}
            """,
        )
    ]
)

formatting_chain = formatting_prompt | llm


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
    docs = None
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

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I Will mak a quiz from Wikipedia article or files you upload to test 
    your knowledge and help you study.
    
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:

    start = st.button("Generate Quiz")

    if start:
        questions_response = questions_chain.invoke(docs)
        formatting_response = formatting_chain.invoke(
            {"context": questions_response.content}
        )
        st.write(formatting_response.content)
