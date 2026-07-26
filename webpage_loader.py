# WebBaseLoader
# WebBaseLoader is a document loader in LangChain used to load and extract text content from web pages (URLs).
# It uses BeautifulSoup under the hood to parse HTML and extract visible text.
# When to Use:
# For blogs, news articles, or public websites where the content is primarily text-based and static.
# Limitations:
# Doesn't handle JavaScript-heavy pages well (use Selenium URLLoader for that).
# Loads only static content (what's in the HTML, not what loads after the page renders).
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

url='https://www.amazon.in/hz/mobile/mission/?_encoding=UTF8&p=9WgdTJdfuB0VNF7jZPXNQ3%2ByKrLaNMODyRv4ElbGcS5fbdOsqLoqg74PowqzjyqVn2%2FOhJZn57gjh3%2FH1eOiM9%2FQwS0S%2FXqyh9tcpm3otiUvIntKxplxm2VZiWlWqbEpRuIR3G9lKMviGrg50UCEd8q5DazrStjr3cDCknYZ3lCEPw5gO7h0HYWf%2B%2F336swY1I22lv4toYcvLTYaMu02rHFdOx39CY%2BgfAvg5OsvB8%2BLXuIM6tYMOa2N6l6REHLVPMsdQSgQT8xrJCdoOxHWYzuCqEr2hOcNVI0ftPtVzbB1l4HnMglqUxKu7cVQscRbUfetptXv%2B7d4KZ%2Fq5qRUPBmGVfDXibKw0CAfDFxDWEsJj7tVuhAFXsgqH8oi3%2BHnzx77Ss%2FhGhtnv84dXucOWACI8m8RBDdRGPCLEAlDPzhB3dtncMdBbXotMljRSk3mkw6iJkZKvoWE080oRGt2j320r6l7%2F5U844aNnqEjKNP7IHyd94D3agj6tmm%2FGgHIaHN2%2BKQXbHSFbOaofpSHIx0nTtpVkFg3B1R0UoqlPUknTiYp6kekFIx2%2FxInu7e9I3G5hryIW%2FQ%3D&pd_rd_w=PRrXo&content-id=amzn1.sym.3b239ffd-5a7d-4d5d-a2af-b2b512da4a2f%3Aamzn1.symc.7cb1f3f5-2abf-40fe-96b7-396919f9e492&pf_rd_p=3b239ffd-5a7d-4d5d-a2af-b2b512da4a2f&pf_rd_r=Y0054TX6X8QWDJ24YT0X&pd_rd_wg=Homuo&pd_rd_r=db538254-dfa2-49d8-b5ae-ab677086a068&ref_=pd_hp_d_atf_ci_mcx_mi_' 


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)


prompt= PromptTemplate(
    template='Answer the question {question} based on following texts {texts}',
    input_variables=['question','texts']
)

parser=StrOutputParser()
loader= WebBaseLoader(url)
docs= loader.load()
chain= prompt | model | parser 


result=chain.invoke({'question':'What is the name of the procuct?', 'texts':docs[0].page_content})
print(result)