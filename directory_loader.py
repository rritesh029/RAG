from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader=DirectoryLoader(
    path='Space',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
docs=loader.lazy_load()#use lazy loading in case of huge number of files or documents

for documents in docs:
    print(documents.metadata)

