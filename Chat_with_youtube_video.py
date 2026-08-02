# can be able to chat about any youtube video 
# like summary of this video, does this video have information about AI
# can be able to chat about any youtube video
# like summary of this video, does this video have information about AI

from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os

os.system("cls")
load_dotenv()

# -------------------------
# Fetch Transcript
# -------------------------

video_id = "B6NVvtIz9_Q"

api = YouTubeTranscriptApi()

transcript = api.fetch(
    video_id,
    languages=["hi"]
)

text = " ".join(snippet.text for snippet in transcript)

# -------------------------
# Translate
# -------------------------

translator = GoogleTranslator(
    source="hi",
    target="en"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100
)

chunks = splitter.split_text(text)


def translate_chunk(chunk):
    return translator.translate(chunk)


with ThreadPoolExecutor(max_workers=5) as executor:
    english_chunks = list(executor.map(translate_chunk, chunks))

# -------------------------
# Documents
# -------------------------

documents = []

for i, english in enumerate(english_chunks, start=1):
    documents.append(
        Document(
            page_content=english,
            metadata={
                "video_id": video_id,
                "chunk_number": i
            }
        )
    )

# -------------------------
# Embeddings
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

# -------------------------
# LLM
# -------------------------

model = ChatOpenAI(
    model="openai/gpt-5.6-luna",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# -------------------------
# Retriever
# -------------------------

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=model
)

query = "Give me short summary of this video in one paragraph"

retrieved_docs = retriever.invoke(query)

# -------------------------
# Build Context
# -------------------------

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

# -------------------------
# Prompt
# -------------------------

prompt = PromptTemplate(
    template="""
You are an assistant.

Answer ONLY from the given context.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke(
    {
        "context": context,
        "question": query
    }
)

print(response)




    
    
    

    