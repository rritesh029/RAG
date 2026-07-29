# it works page by page, it will create the list of document object
# eg: if a pdf has 25 page then it ceate a list of 25 documents with its page content and its meta data
# it internally use PyPDF (not good for scanned pdf or complex layouts)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

spilliter= CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

loader =PyPDFLoader('Voyager_1_and_2.pdf')
docs=loader.load()

result = spilliter.split_documents(docs)
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
print(result[0].page_content)