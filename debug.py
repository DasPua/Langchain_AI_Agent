import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task = "text_generation"
# )

# model = ChatHuggingFace(llm = llm)

# result = model.invoke("how is the weather today ")

# print(result.content)


# import os
# from dotenv import load_dotenv

# load_dotenv()

# print("HF_TOKEN:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))


from search import url_searching
from scrapper import scrapper_tool,Scrapper_func

collected_urls = url_searching("find the top dsa courses")

print(scrapper_tool.invoke(collected_urls))

# print(Scrapper_func(collected_urls))