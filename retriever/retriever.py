from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv 

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task = "text_generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# model = ChatHuggingFace(llm)

# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def retriever(urls):

    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    loader = WebBaseLoader()
    
    loader.web_path = urls

    docs = loader.load()

    document = RecursiveCharacterTextSplitter(
        chunk_size = 100, chunk_overlap = 20
    ).split_documents(docs)

    vector = FAISS.from_documents(document, model)

    ret = vector.as_retriever()
    
    return ret

# result = retriever.invoke("how to use langchain")

# print(result)


