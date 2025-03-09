from langchain.agents import AgentExecutor, initialize_agent, create_tool_calling_agent,create_react_agent
from retriever import retriever_tool
from langchain.schema.runnable import RunnableLambda
from search import search
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain import hub
import os

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text_generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = llm
# prompt = hub.pull("hwchase17/openai-functions-agent") #this is used to make prompt templates for the agent i.e. check using prompt.messages


prompt_template = """You are an AI assistant that helps users with their queries using available tools.
You have access to the following tools:
{tools}
Tool names: {tool_names}

Follow these instructions when answering:
1. Begin by outlining your thought process starting with "Thought:".
2. If a tool is needed, output an "Action:" line with the tool name and input.
3. After processing, conclude with a "Final Answer:" line.
4. Once "Final Answer:" is provided, do not output any further text.
5. End your output with the marker "<END>" on a new line.

Scratchpad for intermediate thoughts:
{agent_scratchpad}

Question: {input}
Your response:"""

prompt = PromptTemplate.from_template(prompt_template)

tools = [search , retriever_tool]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)


# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     verbose=True
# )

# llm_with_tools = RunnableLambda(lambda x: llm.invoke(x))  # Make HuggingFaceEndpoint callable

# agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

agent_executor.invoke({"input": "how can langsmith help with testing?"})
