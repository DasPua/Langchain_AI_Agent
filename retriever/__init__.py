from retriever import retriever
from langchain.tools.retriever import create_retriever_tool


retriever_tool  = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)