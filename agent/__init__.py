from langchain.agents import create_tool_calling_agent
from .weather_agent import model, prompt
from ..search import search
from ..retriever import retriever_tool

tools = [search, retriever_tool]

agent = create_tool_calling_agent(model, tools, prompt)