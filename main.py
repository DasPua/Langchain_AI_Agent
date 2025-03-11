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


from search import url_search

collected_urls = []

for output in url_search.invoke("dsa courses"):
    collected_urls.append(output['url'])

for url in collected_urls:
    print(url)