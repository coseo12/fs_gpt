# NOTE

- 진행 중...(6%)

## Open AI를 위한 요구사항

- [x] Plus 유료 결제
- [x] API Key (유료)
- [x] 반드시 사용 한계 설정 필요 ( 요금 폭탄 주의 )

# 1. 기본 설정

VSCode 기준 필수 플러그인

[Python language support](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
[Jupyter notebook support](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

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

# 2. 랭체인

[랭체인 공식 문서](https://python.langchain.com/v0.1/docs/get_started/quickstart/)
[OpenAI 공식 문서](https://platform.openai.com/docs/overview)

## 2-1. LLM and Chat Models

기본적으로 여러가지 모델들로 작업하기 좋은 인터페이스를 가지고 있으며 각 모델들은 서로 다른 기업에서 제공되고 또한 서로 다른 차이점을 지니고 있지만 랭체인을 사용하면 모든 모델에 호환되는 계층을 사용할 수 있습니다.

[Open AI Models](https://platform.openai.com/docs/models)

### LLM 호출

간단하게 LLM 과 Chat Models 를 호출해보겠습니다.
이 둘은 텍스트를 Predict 할 수 있습니다.

```py
from langchain.llms.openai import OpenAI # LLM
from langchain.chat_models import ChatOpenAI # Chat model

llm = OpenAI()
chat = ChatOpenAI()

a = llm.predict("How many planets are in the solar system?")
b = chat.predict("How many planets are in the solar system?")

a, b
```

## 2-2. Predict Messages

Chat model은 대화에 최적화 되어 있는데 질문을 받을 수 있을 뿐만 아니라 대화를 할 수 있습니다.
즉 메시지의 묶음이라는 의미이며, 상대로서 대화의 맥락에 맞게 대답할 수 있습니다.
Message들을 Predict 해보겠습니다.

```py
from langchain.chat_models import ChatOpenAI
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

---

## 2-3. Prompt Templates

## 2-4. OutputParser and LCEL

## 2-5. Chaining Chains

# 3. 모델

## 3-1. FewShotPromptTemplate

## 3-2. FewShotChatMessagePromptTemplate

## 3-3. 길이 기반 예제 선택기

## 3-4. 직렬화 및 구성

## 3-5. 캐싱

## 3-6. 직렬화

# 4. 메모리

## 4-1. 대화버퍼 메모리

## 4-2. 대화버퍼창 메모리

## 4-3. 대화 요약 메모리

## 4-4. 대화 요약 버퍼 메모리

## 4-5. 대화 KGMemory

## 4-6. LLMChain의 메모리

## 4-7. 채팅 기반 메모리

## 4-8. LCEL 기반 메모리

# 5. 레그

## 5-1. 데이터 로더 및 분할기

## 5-2. 틱토큰

## 5-3. 벡터

## 5-4. 벡터 저장소

## 5-5. 랭스미스

## 5-6. 검색 QA

## 5-7. LCEL 체인 관련 내용

## 5-8. 맵 축소 LCEL cpdls

# 6. 문서 GPT

## 6-1. 매직

## 6-2. 데이터 흐름

## 6-3. 다중 페이지

## 6-4. 채팅 페이지

## 6-5. 문서 업로드

## 6-6. 채팅 기록

## 6-7. 체인

## 6-8. 스트리밍

# 7. 프라이빗 GPT

## 7-1. HuggingFaceHub

## 7-2. HuggingFacePipeline

## 7-3. GPT4

## 7-4. 존재하다

## 7-5. 결론

# 8. 퀴즈 GPT

## 9-1. WikipediaRetriever

## 9-2. GPT-Turbo

## 9-3. 질문 프롬프트

## 9-4. 포멧터 프롬프트

## 9-5. 출력 파서

## 9-6. 캐싱

## 9-7. 채점 질문

## 9-8. 함수 호출

## 9-9. 결론

# 10. 사이트 GPT

## 10-1. AsyncChromiumLoader

## 10-2. 사이트맵 로더

## 10-3. 파싱 기능

## 10-4. 맵 재순위 체인

## 10-5. 맵 리랭크 체인 2부

## 10-6. 코드 챌린지

# 11. 회의 GPT

## 11-1. 오디오 추출

## 11-2. 오디오 자르기

## 11-3. 속삭임 대본

## 11-4. 업로드 UI

## 11-5. 체인 개선 계획

## 11-6. 체인 다듬기

## 11-7. Q&A 탭

# 12. 투자자 GPT

## 12-1. 첫번째 에이전트

## 12-2. 에이전트의 작동 방식

## 12-3. 제로샷 ReAct 에이전트

## 12-4. OpenAI 기능 에이전트

## 12-5. 검색도구

## 12-6. 주식 정보 도구

## 12-7. 에이전트 프롬프트

## 12-8. SQLDatavase Toolkit

# 13. 세프 GPT

## 13-1. CustomGPT 생성

## 13-2. FastAPI 서버

## 13-3. GPT 작업

## 13-4. API 키 인증

## 13-5. OAuth

## 13-6. 솔방울

## 13-7. 세프 API

## 13-8. 코드 첼린지

# 14. 어시스턴트 API

## 14-1. 도우미의 작동 방식

## 14-2. 도우미 만들기

## 14-3. 보조 도구

## 14-4. 스레드 실행

## 14-5. 보조 작업

## 14-6. 코드 챌린지

## 14-7. RAG 도우미

# 15. AzureGPT 및 AWS BEDROCK

## 15-1. AWS BEDROCK

## 15-2. AWS IAM

## 15-4. BEDROCKCHAT

## 15-5. AzureChatOpenAI

# 16. CrewAI

## 16-1. 설정

## 16-2. 승무원, 에이전트 및 작업

## 16-3. 세프크루

## 16-4. 컨텐츠 팜 팀

## 16-5. Pydantic 출력

## 16-6. 비동기 유튜버 제작진

## 16-7. 사용자 정의 도구

## 16-8. 주식 시장 직원
