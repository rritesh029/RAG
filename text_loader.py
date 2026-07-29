from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter

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

spilliter= CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0, #chunk size 100 rakhne se jasie hi 100 count hoga to word v split ho jata hai, eg: agar 100th char kisi word 'Ritesh' k 't' pr aaya to Ritesh split ho jyega so overlap is useful (10-20% of chunk size)
    separator=''
)
spillited_result=spilliter.split_text(docs[0].page_content)
print(spillited_result)

chain= prompt | model | parser 


result = chain.invoke({
    "poem": docs[0].page_content
})

# print(result)

