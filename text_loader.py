from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# model1 = ChatGroq(
#     model="openai/gpt-oss-120b"
# )

prompt= PromptTemplate(
    template='In one line, tell me what is this poem about?-\n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()
loader = TextLoader('fifa.txt',encoding='utf-8')
docs =loader.load()
# print(type(docs))
# print(len(docs))
# print(docs[0])

# print(docs[0].page_content)
# print(docs[0].metadata)

chain= prompt | model | parser 


result = chain.invoke({
    "poem": docs[0].page_content
})

print(result)

# chain1= prompt | model1 | parser 


# result1 = chain1.invoke({
#     "poem": docs[0].page_content
# })

# print(result1)