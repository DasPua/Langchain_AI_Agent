from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text_generation",
)

llm = ChatHuggingFace(llm=endpoint)

def stringtolist(text:str):
    
    ls = [f"{"item"}" for item in text.split(',')[:-1]]
    
    return ls

def Scrapper_func(urls_string):
    
    urls = stringtolist(urls_string)
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    
    transformer = BeautifulSoupTransformer()
    transformed_docs = transformer.transform_documents(docs, tags_to_extract=["span"])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, 
        chunk_overlap=0
    )
    
    splits = splitter.split_documents(transformed_docs)
    
    template = PromptTemplate(
        template = """
        Extract the following information from the text:
        1. Title of the webpage
        2. A brief summary of the content
        
        Text: {content}
        
        Return the information in this format:
        Title: [extracted title]
        Summary: [extracted summary]
        """,
        input_variables=["content"]
    )
    # prompt = ChatPromptTemplate.from_template("""
    # Extract the following information from the text:
    # 1. Title of the webpage
    # 2. A brief summary of the content
    
    # Text: {content}
    
    # Return the information in this format:
    # Title: [extracted title]
    # Summary: [extracted summary]
    # """)
    
    chain = LLMChain(llm=llm, prompt=template)
    print(transformed_docs)
    
    if splits:
        return chain.invoke({"content": splits[0].page_content})
    return "No content found"

urls = "https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN:en,https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(Scrapper_func(urls))