# NOTE

- 진행 중...(12%)

## Open AI를 위한 요구사항

- [x] Plus 유료 결제
- [x] API Key (유료)
- [x] 반드시 사용 한계 설정 필요 ( 요금 폭탄 주의 )

# 1. 기본 설정

VSCode 기준 필수 플러그인

- [Python language support](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter notebook support](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## 1-1. 가상환경 (Python 3.12.4)

### 정의

파이썬 가상환경(Python virtual environment)은 특정 프로젝트에서 필요한 패키지와 라이브러리들을 독립적으로 관리할 수 있게 해주는 도구입니다. 가상환경의 주요 용도는 다음과 같습니다.

- 의존성 관리: 서로 다른 프로젝트에서 서로 다른 버전의 라이브러리나 패키지가 필요할 때, 각 프로젝트가 독립적으로 해당 버전의 패키지를 사용할 수 있게 합니다. 예를 들어, 프로젝트 A에서는 Django 3.0이 필요하고, 프로젝트 B에서는 Django 4.0이 필요한 경우 가상환경을 사용하면 각 프로젝트에서 필요한 버전의 Django를 독립적으로 설치하고 사용할 수 있습니다.

- 시스템 파이썬 환경 보호: 시스템 전체에 영향을 미치지 않고, 특정 프로젝트에만 필요한 패키지들을 설치할 수 있습니다. 이를 통해 시스템 파이썬 환경을 깨끗하게 유지할 수 있습니다.

- 프로젝트 간의 충돌 방지: 여러 프로젝트가 동일한 패키지를 필요로 할 때, 각 프로젝트가 필요한 특정 버전의 패키지를 설치하고 관리할 수 있습니다. 이는 프로젝트 간의 의존성 충돌을 방지합니다.

- 재현 가능한 환경 제공: 특정 프로젝트의 개발 환경을 다른 개발자나 배포 환경에서 재현할 수 있도록 돕습니다. 가상환경을 사용하면 같은 패키지 버전을 설치하여 동일한 환경을 구축할 수 있습니다.

- 배포 환경과의 일치: 개발 환경과 배포 환경을 일치시켜, 배포 시 발생할 수 있는 문제를 줄입니다. 가상환경을 사용하면 로컬 개발 환경과 서버 환경을 쉽게 동기화할 수 있습니다. 요약하면, 파이썬 가상환경은 프로젝트별로 독립된 패키지 관리와 의존성 관리를 가능하게 하여, 개발 과정에서의 효율성과 안정성을 높이는 데 중요한 역할을 합니다.

### 가상환경 생성

여기서는 파이썬 3.3.x 이후 버전에서 표준 모듈로 제공하는 가상환경 venv 를 사용하겠습니다.

```zsh
python -m venv ./env
```

여기서 env는 가상환경의 이름입니다. 원하는 이름으로 변경할 수 있습니다.

### 가상환경 활성화

```zsh
# macOS
source env/bin/activate

# Windows
env\Scripts\activate
```

### 가상환경 비활성화

```zsh
deactivate
```

### 가상환경 삭제

가상환경을 더 이상 사용하지 않을 경우, 가상환경 디렉토리를 삭제하면 됩니다.

```zsh
rm -rf env
```

### 패키지 설치

가상환경이 활성화 된 상태에서 필요한 패키지를 설치합니다.

```zsh
pip install 패키지명
```

### 패키지 목록 정의

requirements.txt 프로젝트에서 필요한 패키지들의 목록과 해당 버전을 명시한 텍스트 파일입니다. 이 파일은 프로젝트의 의존성을 관리하고 다른 환경에서 동일한 패키지 구성을 재현하는 데 사용됩니다.

```zsh
pip install -r requirements.txt
```

### 환경변수 파일 설정

```json
// OPEN AI API KEY
OPENAI_API_KEY="sk-...n"
```

## 1-2. Jupyter Notebook

### 정의

Jupyter Notebook은 대화형 컴퓨팅 환경으로, 특히 데이터 과학, 머신러닝, 데이터 분석, 학술 연구 및 교육에서 널리 사용됩니다. Jupyter Notebook의 주요 용도는 다음과 같습니다.

- 데이터 분석 및 시각화: 데이터를 로드하고, 정제하고, 분석하고, 시각화하는 모든 과정을 한 곳에서 수행할 수 있습니다. pandas, matplotlib, seaborn 등의 라이브러리와 함께 사용하면 강력한 데이터 분석 도구로 활용할 수 있습니다.

- 데이터 과학 및 머신러닝: 데이터 전처리, 모델링, 평가, 예측 등 데이터 과학과 머신러닝의 전체 워크플로우를 처리할 수 있습니다. scikit-learn, TensorFlow, PyTorch 등과 통합하여 머신러닝 모델을 개발하고 실험할 수 있습니다.
- 대화형 코드 실행: 코드를 작성하고 즉시 실행 결과를 확인할 수 있는 대화형 환경을 제공합니다. 이는 실험적 작업이나 코드 디버깅에 매우 유용합니다.

- 교육 및 학습 자료 제작: 교육 자료, 튜토리얼, 강의 노트를 작성하는 데 유용합니다. 코드, 텍스트 설명, 수식, 이미지 등을 하나의 문서에 포함할 수 있어 이해하기 쉽고 직관적인 학습 자료를 만들 수 있습니다.

- 보고서 및 논문 작성: 데이터 분석 결과를 설명하는 보고서나 학술 논문을 작성하는 데 사용할 수 있습니다. 마크다운과 LaTeX을 지원하여 텍스트, 수식, 코드, 결과를 통합한 문서를 작성할 수 있습니다.

- 협업: Jupyter Notebook을 사용하면 다른 개발자나 연구자와 쉽게 협업할 수 있습니다. 노트북 파일(.ipynb)을 공유하여 코드와 분석 결과를 공유하고, 공동 작업할 수 있습니다.

- 데이터 시각화 및 대시보드 생성: Jupyter Notebook을 통해 데이터 시각화를 쉽게 수행할 수 있으며, 대시보드를 생성하여 실시간 데이터 모니터링 및 분석을 할 수 있습니다.

- 재현 가능한 연구: 연구 결과를 재현할 수 있는 환경을 제공하여, 연구의 신뢰성을 높입니다. Jupyter Notebook을 사용하면 코드와 결과를 함께 저장하여 언제든지 동일한 결과를 재현할 수 있습니다. Jupyter Notebook은 코드, 텍스트, 시각화를 통합하여 사용자 친화적인 환경을 제공함으로써, 데이터 과학과 관련된 다양한 작업을 효율적으로 수행할 수 있게 합니다.

### Jupyter Notebook 생성

1. notebook.ipynb 파일을 루트에 생성합니다. (파일이름은 자유롭게 설정합니다.)
2. Select Kernel을 선택하여 현재 파이썬 가상환경으로 접근하도록 경로를 설정합니다.

# 2. LANGCHAIN

- [랭체인 공식 문서](https://python.langchain.com/v0.1/docs/get_started/quickstart/)
- [OpenAI 공식 문서](https://platform.openai.com/docs/overview)

## 2-1. LLM and Chat Models

기본적으로 여러가지 모델들로 작업하기 좋은 인터페이스를 가지고 있으며 각 모델들은 서로 다른 기업에서 제공되고 또한 서로 다른 차이점을 지니고 있지만 랭체인을 사용하면 모든 모델에 호환되는 계층을 사용할 수 있습니다.

- [Open AI Models](https://platform.openai.com/docs/models)

간단하게 LLM 과 Chat Models 를 호출해보겠습니다.
이 둘은 텍스트를 Predict 할 수 있습니다.

```py
from langchain_openai import OpenAI, ChatOpenAI # LLM, Chat model

llm = OpenAI()
chat = ChatOpenAI()

a = llm.invoke("How many planets are in the solar system?")
b = chat.invoke("How many planets are in the solar system?")

a, b
```

## 2-2. Invoke (Predict Messages)

Chat model은 대화에 최적화 되어 있는데 질문을 받을 수 있을 뿐만 아니라 대화를 할 수 있습니다.
즉 메시지의 묶음이라는 의미이며, 상대로서 대화의 맥락에 맞게 대답할 수 있습니다.
Message들을 invoke 해보겠습니다.

```py
from langchain_openai import ChatOpenAI
# HumanMessage - 인간이 작성하는 메시지
# AIMessage - AI에 의해서 보내지는 메시지
# SystemMessage - LLM에 설정들을 제공하기 위한 Message
from langchain.schema import HumanMessage, AIMessage, SystemMessage

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
)

messages = [
    SystemMessage(content="You are a geography expert. And you only reply in Italian."),
    AIMessage(content="Ciao, mi chiamo Paolo!"),
    HumanMessage(content="What is the distance between the Mexico and Thailand. Also, what is your name?"),
]

chat.invoke(messages)
```

## 2-3. Prompt Templates

prompt란 LLM 과 의사소통할 수 있는 방법입니다. prompt의 성능이 좋다면 LLM의 답변도 좋아집니다.
모든 웹사이트들이 상황에 맞는 뛰어난 성능의 prompt를 제작하는데 많은 노력을 기울입니다.
Langchain은 prompt를 공유하기 위한 커뮤니티를 만들고 있습니다. 이를 이용하여 많은 사용자들이 prompt를 공유할 수 있습니다. 많은 유틸리티 들이 prompt를 위해 존재합니다.

간단하게 문자열을 통한 predict를 실행하는 예제를 작성해 보겠습니다.

```py
from langchain_openai import ChatOpenAI
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
)

template = PromptTemplate.from_template("What is the distance between {country_a} and {country_b}")

prompt = template.format(country_a="Mexico", country_b="Thailand")

chat.invoke(prompt)
```

이번에는 메시지를 통한 invoke를 실행하는 예제를 작성해 보겠습니다.

```py
from langchain_openai import ChatOpenAI
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
)

messages = [
    SystemMessagePromptTemplate.from_template("You are a geography expert. And you only reply in {language}."),
    AIMessagePromptTemplate.from_template("Ciao, mi chiamo {name}!"),
    HumanMessagePromptTemplate.from_template("What is the distance between the {country_a} and {country_b}. Also, what is your name?")
]

template = ChatPromptTemplate.from_messages(messages)

prompt = template.format_messages(language="Italian", name="Paolo", country_a="Mexico", country_b="Thailand")

chat.invoke(prompt)
```

## 2-4. OutputParser and LCEL

OutputParser는 LLM의 응답(Response)을 다양한 형태로 변형을 하기 위해서 사용합니다.
LCEL(langchain expression language)은 복잡할 수도 있는 코드를 간결하게 만들 수 있습니다. 그리고 다양한 template와 LLM 호출, 그리고 서로 다른 응답(Response)를 함께 사용할 수 있습니다.

첫번째로 OutputParser 예제로 간단하게 응답을 list로 변환해보겠습니다.

```py
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
)

# 문자열 출력을 파싱하는 BaseOutputParser 확장하는 커스텀 OutputParser
class CommaOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        items = text.strip().split(",")
        return list(map(str.strip, items))

p = CommaOutputParser()

messages = [
    SystemMessagePromptTemplate.from_template("You are a list gernerating machine. Everything you are asked will be answered with a comma separated list of max {max_items} in lowercase. Do Not reply with else."),
    HumanMessagePromptTemplate.from_template("{question}")
]
template = ChatPromptTemplate.from_messages(messages)
prompt = template.format_messages(max_items=10, question="What are the colors?")
res = chat.invoke(prompt)
p.parse(res.content)
```

결과는 단순하지만 실행하는 코드는 너무 복잡합니다. 이것을 단순화하기 위해 Chaining 하도록 변경해 보겠습니다.

```py
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
)

# 문자열 출력을 파싱하는 BaseOutputParser 확장하는 커스텀 OutputParser
class CommaOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        items = text.strip().split(",")
        return list(map(str.strip, items))

messages = [
    SystemMessagePromptTemplate.from_template("You are a list gernerating machine. Everything you are asked will be answered with a comma separated list of max {max_items} in lowercase. Do Not reply with else."),
    HumanMessagePromptTemplate.from_template("{question}")
]

# ✨Chaining✨
chain = template | chat | CommaOutputParser()
chain.invoke({
    "max_items":10,
    "question":"What are the colors?",
})
```

## 2-5. Chaining Chains

- [Expression Language/interface](https://python.langchain.com/v0.1/docs/expression_language/interface/)

Chaining과 LCEL(langchain expression language)에 대하여 좀 더 깊게 알아보도록 하겠습니다.
이전 Chaining과 코드를 살펴보겠습니다.

```py
chain = template | chat | CommaOutputParser()
chain.invoke({
    "max_items":10,
    "question":"What are the colors?",
})
```

우리는 현재 prompt와 chat model 그리고 OutputParser를 사용하고 있습니다. 이 외에도 다른 타입들은 위에 공식문서 링크를 참조해주세요(밑에서 다른 타입들도 사용하게 됩니다.)

실행 시 첫번째 template.format_messages 즉 prompt 명령이 실행 됩니다. 이로 인하여 그 값을 Dictionary형태로 전달하고 있습니다. 이로 인하여 Dictionary형태의 매개변수를 전달하고 있습니다. 첫번째 실행의 결과는 prompt value를 받게 됩니다.
두번째로 chat model로 첫번째 실행결과(prompt value)와 함께 이동합니다. chat model은 prompt value를 매개로 실행되며 그 실행 결과를 String 형태로 받게 됩니다.
세번째로 OutputParser로 두번째 실행결과(String)와 함께 이동합니다. OutputParser는 우리가 원하는 형태로 문자열을 가공하여 마지막 결과를 출력하게 될 것입니다.

이제 Chain들을 서로 Chaining 하는 예제를 작성해 보겠습니다.

```py
from langchain_openai import ChatOpenAI
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
    streaming=True, # streaming 옵션을 활성화하여 대화형 모드로 설정
    callbacks=[StreamingStdOutCallbackHandler()], # 콜백 함수를 설정
)

chef_message =  [
    SystemMessagePromptTemplate.from_template("You are a world-class international chef. You create easy to fllow recipies for any type of cuisine with easy to find ingredients."),
    HumanMessagePromptTemplate.from_template("I want to cook {cuisine} food.")
]

chef_prompt = ChatPromptTemplate.from_messages(chef_message)

chef_chain = chef_prompt | chat

veg_chef_message =  [
    SystemMessagePromptTemplate.from_template("You are a vegetarian chef specialized on marking tranditional recipies vegetarian. You find alternatibe ingredients and explain their preparation. You don't redically modify the recipe. If there is no alternative for a food just say you don't know how to replace it."),
    HumanMessagePromptTemplate.from_template("{recipe}")
]

veg_chef_prompt = ChatPromptTemplate.from_messages(veg_chef_message)

veg_chef_chain = veg_chef_prompt | chat

# ✨RunnableMap✨사용
final_chain = {"recipe": chef_chain} | veg_chef_chain

final_chain.invoke({
    "cuisine":"indian",
})
```

# 3. MODEL I/O

Langchain에는 다양한 Model I/O가 존재합니다. 이는 다른 모든 언어 모델들과 인터페이스 할 수 있는 빌딩 블록을 제공합니다.

![Model Image](./images/model.png)

[Components/Modules](https://python.langchain.com/v0.1/docs/modules/)

## 3-1. FewShotPromptTemplate

우리는 Prompt Template를 통하여 메시지의 유효성 확인하고 또한 저장 및 불러오기를 할 수 있습니다. 규모가 있는 LLM을 만들기 시작할 때 Prompt는 매우 중요합니다.
기본적으로 Fewshot 은 Model 들에게 예제들을 준다는 뜻과 같습니다. 이는 더 좋은 대답을 할 수 있도록 하는 예제들을 제공하는 것입니다.
예를 들어 구체적으로 대답하는 AI Model이 필요하다고 가정했을 시 어떻게 대답해야 하는 지에 대한 예제를 AI Model에게 제공하였을 때 Prompt를 사용해서 어떻게 대답해야 하는지 알려주는 것보다 더 좋습니다. 왜냐하면 모델은 텍스트를 만들기 때문에 Prompt로 명령을 하는 것보다 어떻게 대답해야 하는지 예제를 제공해주는 것이 더 좋은 방법입니다. 이것이 FewShotPromptTemplate이 하는 일이며 이를 통하여 예제를 형식화 할 수 있습니다.
또한 예제들이 데이터베이스에 있을 수도 있기 때문에 이런 대화 기록 같은 것들을 데이터베이스에서 가져와서 FewShotPromptTemplate이 사용하여 형식화 시켜주면 더 빠르게 잘 만들 수 있습니다.

이제 간단한 예제를 작성해보겠습니다.

```py
from langchain_openai import ChatOpenAI
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
    streaming=True, # streaming 옵션을 활성화하여 대화형 모드로 설정
    callbacks=[StreamingStdOutCallbackHandler()], # 콜백 함수를 설정
)

# 모델에게 전달하는 답변 예제
examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    }]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt, # Prompt 방식
    examples=examples, # 답변 예제
    suffix="Human: Wat do you know about {country}?", # 모든 형식화된 예제 마지막 내용
    input_variables=["country"] # suffix 입력 변수 (유효성 검사)
)

chain = prompt | chat

chain.invoke({
    "country":"Germany",
})
```

## 3-2. FewShotChatMessagePromptTemplate

이제 단순 문자열 형태가 아닌 메시지 형태의 FewShotChatMessagePromptTemplate를 작성해 보겠습니다.

```py
from langchain_openai import ChatOpenAI
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import PromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
    streaming=True, # streaming 옵션을 활성화하여 대화형 모드로 설정
    callbacks=[StreamingStdOutCallbackHandler()], # 콜백 함수를 설정
)

# 모델에게 전달하는 답변 예제
examples = [
    {
        "country": "France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "country": "Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "country": "Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    }]


example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("What do you know about {country}?"),
    AIMessagePromptTemplate.from_template("{answer}"),
]
)

prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, # Prompt 방식
    examples=examples, # 답변 예제
)

final_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a geography expert, you give short answers."),
    prompt,
    HumanMessagePromptTemplate.from_template("What do you know about {country}?")
])

chain = final_prompt | chat

chain.invoke({
    "country":"Germany",
})
```

## 3-3. LengthBasedExampleSelector

여기에서는 동적으로 예제들을 선택할 수 있는 방법에 대해 알아보겠습니다. 상황에 따라서는 많은 예제들이 존재하고 어느정도 예제들을 골라서 Prompt에 허용할 것인가에 대해 정의를 해야합니다. 이유는 많은 Prompt는 더 큰 비용을 지불해야 하며 비용이 존재하더라도 모델에 알맞은 양이 존재합니다.

간단한 길이를 조절하는 기본 예제를 작성해 보겠습니다.

```py
from langchain_openai import ChatOpenAI
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import PromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.example_selector import LengthBasedExampleSelector

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
    streaming=True, # streaming 옵션을 활성화하여 대화형 모드로 설정
    callbacks=[StreamingStdOutCallbackHandler()], # 콜백 함수를 설정
)

# 모델에게 전달하는 답변 예제
examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    }]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")

# 예제 선택
example_selector = LengthBasedExampleSelector(
    examples=examples, # 답변 예제
    example_prompt=example_prompt, # Prompt 방식
    max_length=180 # 최대 길이
)


prompt = FewShotPromptTemplate(
    example_prompt=example_prompt, # Prompt 방식
    example_selector=example_selector, # 답변 선택
    suffix="Human: Wat do you know about {country}?", # 모든 형식화된 예제 마지막 내용
    input_variables=["country"] # suffix 입력 변수 (유효성 검사)
)

prompt.format(country="Brazil")
```

랜덤한 예제를 선택하도록 수정해 보겠습니다.

```py
from langchain_openai import ChatOpenAI
# PromptTemplate - 문자열을 이용한 template 생성
# ChatPromptTemplate - message를 이용하여 template 생성
from langchain.prompts import PromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
# from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector

class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        from random import choice
        return [choice(self.examples)]

chat = ChatOpenAI(
    temperature=0.1, # 모델의 창의성을 조절하는 옵션 (높을 수록 창의적임)
    streaming=True, # streaming 옵션을 활성화하여 대화형 모드로 설정
    callbacks=[StreamingStdOutCallbackHandler()], # 콜백 함수를 설정
)

# 모델에게 전달하는 답변 예제
examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    }]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")

# 예제 선택
example_selector = RandomExampleSelector(
    examples=examples, # 답변 예제
)


prompt = FewShotPromptTemplate(
    example_prompt=example_prompt, # Prompt 방식
    example_selector=example_selector, # 답변 선택
    suffix="Human: Wat do you know about {country}?", # 모든 형식화된 예제 마지막 내용
    input_variables=["country"] # suffix 입력 변수 (유효성 검사)
)

prompt.format(country="Brazil")
```

## 3-4. Serialization and Composition

```py

```

## 3-5. Caching

```py

```

## 3-6. Serialization

```py

```

# 4. MEMORY

## 4-1. ConversationBufferMemory

## 4-2. ConversationBufferWindowMemory

## 4-3. ConversationSummaryMemory

## 4-4. ConversationSummaryBufferMemory

## 4-5. ConversationKGMemory

## 4-6. Memory on LLMChain

## 4-7. Chat Based Memory

## 4-8. LCEL Based Memory

# 5. RAG

## 5-1. Data Loaders and Splitters

## 5-2. Tiktoken

## 5-3. Vectors

## 5-4. Vectors Store

## 5-5. Langsmith

## 5-6. RetrievalQA

## 5-7. Stuff LCEL Chain

## 5-8. Map Reduce LCEL Chain

# 6. DOCUMENT GPT

## 6-1. Magic

## 6-2. Data Flow

## 6-3. Multi Page

## 6-4. Chat Message

## 6-5. Uploading Documents

## 6-6. Chat History

## 6-7. Chain

## 6-8. Streaming

# 7. PRIVATE GPT

## 7-1. HuggingFaceHub

## 7-2. HuggingFacePipeline

## 7-3. GPT4ALL

## 7-4. Ollama

## 7-5. Conclusions

# 8. QUIZ GPT

## 8-1. WikipediaRetriever

## 8-2. GPT4-Turbo

## 8-3. Questions Prompt

## 8-4. Formatter Prompt

## 8-5. Output Parser

## 8-6. Caching

## 8-7. Grading Questions

## 8-8. Function Calling

## 8-9. Conclusions

# 9. SITE GPT

## 9-1. AsyncChromiumLoader

## 9-2. SitemapLoader

## 9-3. Parsing Function

## 9-4. Map Re Rank Chain

## 9-5. Map Re Rank Chain part Two

## 9-6. Code Challenge

# 10. MEETING GPT

## 10-1. Audio Extraction

## 10-2. Cutting The Audio

## 10-3. Whisper Transcript

## 10-4. Upload UI

## 10-5. Refine Chain Plan

## 10-6. Refine Chain

## 10-7. Q&A Tab

# 11. INVEST OR GPT

## 11-1. Your First Agent

## 11-2. How do Agents Work

## 11-3. Zero-shot ReAct Agent

## 11-4. OpenAI Functions Agent

## 11-5. Search Toll

## 11-6. Stock Information Tools

## 11-7. Agent Prompt

## 11-8. SQLDatavase Toolkit

## 11-9. Conclusions

# 12. CHEF GPT

## 12-1. CustomGPT Creation

## 12-2. FastAPI Server

## 12-3. GPT Action

## 12-4. API Key Auth

## 12-5. OAuth

## 12-6. Chef API

## 12-7. Code Challenge

## 12-8. Conclusions

# 13. ASSISTANTS API

## 13-1. How Assistants Work

## 13-2. Creating The Assistants

## 13-3. Assistants Tools

## 13-4. Running A Thread

## 13-5. Assistants Actions

## 13-6. Code Challenge

## 13-7. RAG Assistant

## 13-8. Conclusions

# 14. AzureGPT & AWS BEDROCK

## 14-1. AWS BEDROCK

## 14-2. AWS IAM

## 14-4. BEDROCKCHAT

## 14-5. AzureChatOpenAI

# 15. CrewAI

## 15-1. Setup

## 15-2. Crews, Agents and Tasks

## 15-3. Chef Crew

## 15-4. Content Farm Crew

## 15-5. Pydantic Outputs

## 15-6. Async Youtuber Crew

## 15-7. Custom Tools

## 15-8. Stock Market Crew

## 15-9. Conclusions
