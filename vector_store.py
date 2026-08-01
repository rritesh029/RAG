from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
os.system("cls")

doc1 = Document(
    page_content=(
        "Chennai Super Kings is an IPL franchise based in Chennai. "
        "The team plays its home matches at M. A. Chidambaram Stadium. "
        "Ruturaj Gaikwad is the captain of Chennai Super Kings. "
        "The squad includes players such as MS Dhoni, Sanju Samson, "
        "Shivam Dube and Noor Ahmad."
    ),
    metadata={"team": "Chennai Super Kings"}
)

doc2 = Document(
    page_content=(
        "Mumbai Indians is an IPL franchise based in Mumbai. "
        "The team plays its home matches at Wankhede Stadium. "
        "The squad includes prominent players such as Rohit Sharma, "
        "Hardik Pandya, Suryakumar Yadav and Jasprit Bumrah."
    ),
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content=(
        "Royal Challengers Bengaluru is an IPL franchise based in Bengaluru. "
        "The team plays its home matches at M. Chinnaswamy Stadium. "
        "Rajat Patidar is the captain of Royal Challengers Bengaluru. "
        "The squad includes players such as Virat Kohli, Phil Salt, "
        "Jitesh Sharma, Krunal Pandya and Josh Hazlewood."
    ),
    metadata={"team": "Royal Challengers Bengaluru"}
)

doc4 = Document(
    page_content=(
        "Kolkata Knight Riders is an IPL franchise based in Kolkata. "
        "The team plays its home matches at Eden Gardens. "
        "Ajinkya Rahane is the captain of Kolkata Knight Riders. "
        "The squad includes players such as Rinku Singh, Sunil Narine, "
        "Varun Chakaravarthy and Cameron Green."
    ),
    metadata={"team": "Kolkata Knight Riders"}
)

doc5 = Document(
    page_content=(
        "Rajasthan Royals is an IPL franchise based in Rajasthan. "
        "The squad includes players such as Yashasvi Jaiswal, "
        "Riyan Parag, Ravindra Jadeja, Jofra Archer and Shimron Hetmyer. "
        "Ravindra Jadeja is an experienced Indian all-rounder who contributes "
        "with both batting and spin bowling."
    ),
    metadata={"team": "Rajasthan Royals"}
)

doc6 = Document(
    page_content=(
        "Sunrisers Hyderabad is an IPL franchise based in Hyderabad. "
        "The squad includes players such as Pat Cummins, Travis Head, "
        "Abhishek Sharma, Ishan Kishan and Heinrich Klaasen. "
        "The team has a strong combination of aggressive batting and pace bowling."
    ),
    metadata={"team": "Sunrisers Hyderabad"}
)

doc7 = Document(
    page_content=(
        "Delhi Capitals is an IPL franchise based in Delhi. "
        "The squad includes players such as KL Rahul, Axar Patel, "
        "Mitchell Starc, T. Natarajan and Tristan Stubbs. "
        "Axar Patel is an Indian all-rounder known for left-arm spin "
        "and useful middle-order batting."
    ),
    metadata={"team": "Delhi Capitals"}
)

doc8 = Document(
    page_content=(
        "Punjab Kings is an IPL franchise representing Punjab. "
        "The squad includes players such as Shreyas Iyer, Arshdeep Singh, "
        "Yuzvendra Chahal, Marcus Stoinis and Marco Jansen. "
        "The team combines Indian batting talent with experienced international players."
    ),
    metadata={"team": "Punjab Kings"}
)

doc9 = Document(
    page_content=(
        "Gujarat Titans is an IPL franchise based in Gujarat. "
        "The squad includes players such as Shubman Gill, Rashid Khan, "
        "Jos Buttler, Mohammed Siraj and Sai Sudharsan. "
        "Shubman Gill is a top-order Indian batter known for technically strong batting."
    ),
    metadata={"team": "Gujarat Titans"}
)

doc10 = Document(
    page_content=(
        "Lucknow Super Giants is an IPL franchise based in Lucknow. "
        "Rishabh Pant is the captain of Lucknow Super Giants. "
        "The squad includes players such as Nicholas Pooran, Mitchell Marsh, "
        "Aiden Markram, Mohammad Shami and Ayush Badoni."
    ),
    metadata={"team": "Lucknow Super Giants"}
)

docs = [
    doc1,
    doc2,
    doc3,
    doc4,
    doc5,
    doc6,
    doc7,
    doc8,
    doc9,
    doc10
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store= Chroma(
    embedding_function=embedding_model,
    persist_directory='chroma_db',
    collection_name='sample'
)

ids = [
    "csk", "mi", "rcb", "kkr", "rr",
    "srh", "dc", "pbks", "gt", "lsg"
]

vector_store.add_documents(
    documents=docs,
    ids=ids
)


result = vector_store.get(
    include=["embeddings"]
)

# print(result)

query = """
I like teams with strong all-rounders who can
contribute with both batting and spin bowling.
"""

results = vector_store.similarity_search(
    query=query,
    k=3
)

for doc in results:
    print(doc.metadata["team"])
    print(doc.page_content)
    print("-------------------")