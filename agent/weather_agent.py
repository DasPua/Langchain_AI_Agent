from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain import hub
import os

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text_generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = llm
prompt = hub.pull("hwchase17/openai-functions-agent") #this is used to make prompt templates for the agent i.e. check using prompt.messages



print(prompt)


