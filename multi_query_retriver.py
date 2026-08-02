# it split the user query into multiple related ambiguous query and one retriver will be assigned to each ambiguous query to find the relevant documents and then all output will be merged to one to show as a repsonse
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


import os
os.system("cls")

load_dotenv()
all_docs = [
Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "11"}),
Document (page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "12"}),
Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "13"}),
Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "14"}),
Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "15"}),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
model = ChatOpenAI(
    model="openai/gpt-5.6-luna",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# vector_store= Chroma(
#     embedding_function=embedding_model,
#     persist_directory='chroma_db',
#     collection_name='sample'
# )

vector_store= FAISS.from_documents(documents=all_docs,embedding=embedding_model,)
similarity_retriver=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})


multiquery_retriver= MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k":5}),
    llm=model
) 

query="How to improve energy levels and maintain balance?"

similarity_result= similarity_retriver.invoke(query)
multiquery_retriver_result= multiquery_retriver.invoke(query)
for i, doc in enumerate (similarity_result):
    print(i, doc.page_content)
    print("\n")
print("<------------------------------------->")
    
for i, doc in enumerate(multiquery_retriver_result):
    print(i, doc.page_content)
    print("\n")    