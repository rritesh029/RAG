from langchain_community.retrievers import WikipediaRetriever
import os

os.system("cls")

retriver= WikipediaRetriever(top_k_results=2, lang="en")
query= "the geopolitical history of indian and pakistan from the perspective of a chinese"

docs= retriver.invoke(query)

for i in range(len(docs)):
    doc=docs[i]
    print(i, doc.page_content)