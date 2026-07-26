from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from langchain_core.documents import Document

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)


prompt= PromptTemplate(
    template='Answer the question {question} based on following texts {texts}',
    input_variables=['question','text']
)


df = pd.read_excel("Acct_Statement.xls", engine="xlrd")

docs = [
    Document(page_content=row.to_string())
    for _, row in df.iterrows()
]


interest_docs = [
    doc for doc in docs
    if "interest" in doc.page_content.lower()
]

texts = "\n".join(
    doc.page_content for doc in interest_docs
)


parser=StrOutputParser()
# loader= CSVLoader(file_path='Acct_Statement.xls')
# docs= loader.load()
chain= prompt | model | parser 


result=chain.invoke({'question':'Check the 2nd column, find all the interest rows and sum there corresponding column having column name Deposite Amt and give the result', 'texts':texts})
print(result)