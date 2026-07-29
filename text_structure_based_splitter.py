# Recursive character text splitting technique

# isme phle se separated define kr lete h 
# eg: paragraph k liye '\n\n'
# line change k liye '\n'
# spaces k liye space, represent words
# '' for character, it represent character
# phle para k basis pr chunks banega, fir line k basis pr, fir word and so on.
# for more clarity watch https://www.youtube.com/watch?v=SEWS9P4ODmc&t=954s (time 26mins onwards)


from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt= PromptTemplate(
    template='In one line, tell me what is this poem about?-\n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()
loader = TextLoader('fifa.txt',encoding='utf-8')
docs =loader.load()

splitter= RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0 #chunk size 100 rakhne se jasie hi 100 count hoga to word v split ho jata hai, eg: agar 100th char kisi word 'Ritesh' k 't' pr aaya to Ritesh split ho jyega so overlap is useful (10-20% of chunk size)
   
)
texts=docs[0].page_content
splitted_result=splitter.split_text(texts)
print(len(splitted_result))
print(splitted_result)

chain= prompt | model | parser 


result = chain.invoke({
    "poem": docs[0].page_content
})

# print(result)

