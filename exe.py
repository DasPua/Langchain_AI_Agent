from langchain.agents import AgentExecutor, initialize_agent, create_tool_calling_agent,create_react_agent
from retriever import retriever_tool
from langchain.schema.runnable import RunnableLambda
from search import search,url_searching
from scrapper import scrapper
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain import hub
import os

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text_generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = llm

prompt_template = prompt_template = """You are an AI assistant that helps users with their queries using available tools.
You have access to the following tools:
{tools}
Tool names: {tool_names}

Follow these EXACT formatting instructions:
1. Begin with "Thought: [your reasoning]"
2. Next line must be "Action: [tool_name]"
3. Next line must be "Action Input: [input for the tool]"
4. After receiving tool output, continue with "Thought:"
5. When ready to give final answer, use "Final Answer: [your answer]"

Example of correct format:
Thought: I need to search for information about X
Action: search_tool
Action Input: query about X

Scratchpad for intermediate thoughts:
{agent_scratchpad}

Question: {input}
Your response:"""

prompt = PromptTemplate.from_template(prompt_template)

llm_with_tools = RunnableLambda(lambda x, **kwargs: llm.invoke(x))

tools = [url_searching, scrapper]

agent = create_react_agent(llm=llm_with_tools, tools=tools, prompt=prompt)

# agent = create_tool_calling_agent(llm_with_tools, tools, prompt = prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

agent_executor.invoke({"input": "news about cricket?"})
