# Return only the relevant part
# if a text contains information about two separate thing say A and B and if query send by the user is related to A then it will trim the text for B and then send the response
# How It Works
# 1. Base Retriever (e.g., FAISS, Chroma) retrieves N documents.
# 2. A compressor (usually an LLM) is applied to each document.
# 3. The compressor keeps only the parts relevant to the query.
# 4. Irrelevant content is discarded.

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv


import os
os.system("cls")
load_dotenv()

docs = [
    Document(
        page_content="""
The Grand Canyon is one of the most visited natural wonders in the world.
Photosynthesis is the process by which green plants convert sunlight into energy.
Millions of tourists travel to see it every year.
The rocks date back millions of years.
""",
        metadata={"source": "Doc1"},
    ),

    Document(
        page_content="""
In medieval Europe, castles were built primarily for defense.
The chlorophyll in plant cells captures sunlight during photosynthesis.
Knights wore armor made of metal.
Siege weapons were often used to breach castle walls.
""",
        metadata={"source": "Doc2"},
    ),

    Document(
        page_content="""
Basketball was invented by Dr. James Naismith in the late 19th century.
It was originally played with a soccer ball and peach baskets.
The NBA is now a global league.
""",
        metadata={"source": "Doc3"},
    ),

    Document(
        page_content="""
The history of cinema began in the late 1800s.
Silent films were the earliest form.
Thomas Edison was among the pioneers.
Photosynthesis does not occur in animal cells.
Modern filmmaking involves complex CGI and sound design.
""",
        metadata={"source": "Doc4"},
    ),
]


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

vector_store= FAISS.from_documents(documents=docs, embedding=embedding_model,)

base_retriver= vector_store.as_retriever(search_kwargs={"k":5})

llm= ChatOpenAI(
    model="openai/gpt-5.6-luna",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
compressor= LLMChainExtractor.from_llm(llm)
compression_retriver=ContextualCompressionRetriever(
    base_retriever=base_retriver,
    base_compressor=compressor
)
query="what is photosynthesis?"
compressed_result= compression_retriver.invoke(query)

print("\n")
for i, doc in enumerate(compressed_result):
    print(i, doc.page_content)
    print("\n")