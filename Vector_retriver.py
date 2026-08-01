from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
os.system("cls")

doc1 = Document(
    page_content=(
        "Chennai Super Kings is an IPL franchise based in Chennai. "
        "The team plays its home matches at M. A. Chidambaram Stadium. "
        "MS Dhoni is the legendary wicketkeeper and mentor of the team. "
        "Ravindra Jadeja is one of the world's finest all-rounders, contributing with aggressive batting, left-arm spin bowling, and exceptional fielding. "
        "Shivam Dube is another batting all-rounder who strengthens the middle order."
    ),
    metadata={"team": "Chennai Super Kings"}
)

doc2 = Document(
    page_content=(
        "Mumbai Indians is an IPL franchise based in Mumbai. "
        "The team plays at Wankhede Stadium. "
        "Hardik Pandya is a pace-bowling all-rounder who contributes with explosive batting and fast bowling. "
        "Rohit Sharma is a successful opening batter. "
        "Jasprit Bumrah is regarded as one of the world's best fast bowlers."
    ),
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content=(
        "Royal Challengers Bengaluru is an IPL franchise based in Bengaluru. "
        "The team plays its home matches at M. Chinnaswamy Stadium. "
        "Virat Kohli is the leading batter of the team. "
        "Krunal Pandya is a left-arm spin all-rounder who contributes with both batting and bowling. "
        "Josh Hazlewood leads the fast bowling attack."
    ),
    metadata={"team": "Royal Challengers Bengaluru"}
)

doc4 = Document(
    page_content=(
        "Kolkata Knight Riders is an IPL franchise based in Kolkata. "
        "The team plays its home matches at Eden Gardens. "
        "Sunil Narine is a mystery spinner and a powerful opening batter. "
        "Andre Russell is one of the most destructive pace-bowling all-rounders in T20 cricket. "
        "Rinku Singh is an explosive middle-order batter."
    ),
    metadata={"team": "Kolkata Knight Riders"}
)

doc5 = Document(
    page_content=(
        "Rajasthan Royals is an IPL franchise based in Rajasthan. "
        "The squad includes Ravindra Jadeja, Riyan Parag, Jofra Archer and Shimron Hetmyer. "
        "Ravindra Jadeja is one of India's finest all-rounders who contributes with batting, left-arm spin bowling and outstanding fielding. "
        "Riyan Parag is an emerging batting all-rounder."
    ),
    metadata={"team": "Rajasthan Royals"}
)

doc6 = Document(
    page_content=(
        "Sunrisers Hyderabad is an IPL franchise based in Hyderabad. "
        "The team includes Pat Cummins, Travis Head, Heinrich Klaasen and Abhishek Sharma. "
        "Abhishek Sharma is a batting all-rounder who also bowls left-arm spin. "
        "Pat Cummins contributes with lower-order batting and fast bowling."
    ),
    metadata={"team": "Sunrisers Hyderabad"}
)

doc7 = Document(
    page_content=(
        "Delhi Capitals is an IPL franchise based in Delhi. "
        "The squad includes KL Rahul, Axar Patel, Mitchell Starc and Tristan Stubbs. "
        "Axar Patel is a dependable all-rounder known for economical left-arm spin bowling and useful batting. "
        "KL Rahul is a technically sound top-order batter."
    ),
    metadata={"team": "Delhi Capitals"}
)

doc8 = Document(
    page_content=(
        "Punjab Kings is an IPL franchise representing Punjab. "
        "The team includes Marcus Stoinis, Marco Jansen, Arshdeep Singh and Shreyas Iyer. "
        "Marcus Stoinis is a pace-bowling all-rounder. "
        "Marco Jansen contributes with fast bowling and lower-order batting."
    ),
    metadata={"team": "Punjab Kings"}
)

doc9 = Document(
    page_content=(
        "Gujarat Titans is an IPL franchise based in Gujarat. "
        "The squad includes Shubman Gill, Rashid Khan, Rahul Tewatia and Mohammed Siraj. "
        "Rashid Khan is a world-class leg spinner and an explosive lower-order batter. "
        "Rahul Tewatia is another useful all-rounder capable of finishing matches."
    ),
    metadata={"team": "Gujarat Titans"}
)

doc10 = Document(
    page_content=(
        "Lucknow Super Giants is an IPL franchise based in Lucknow. "
        "The team includes Mitchell Marsh, Nicholas Pooran, Aiden Markram and Rishabh Pant. "
        "Mitchell Marsh is a genuine pace-bowling all-rounder. "
        "Aiden Markram contributes with batting and useful off-spin bowling."
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

# results = vector_store.similarity_search(
#     query=query,
#     k=3
# )

# for doc in results:
#     print(doc.metadata["team"])
#     print(doc.page_content)
#     print("-------------------")

retriver= vector_store.as_retriever(search_kwarga={"k":2})
output_of_query=retriver.invoke(query)

for i in range(len(output_of_query)):
    doc=output_of_query[i]
    print(i, doc.page_content)
    print("\n")