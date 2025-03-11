from retriever.retriever import retriever
from langchain_community.tools import Tool


retriever_tool  = Tool(
    name = "retriever",
    description = "Search and stores the information about the query from from a list of URLs. Input should be a comma-separated string of URLs or a list of URLs.",
    func = retriever
)