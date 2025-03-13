from langchain.agents import AgentExecutor, initialize_agent, create_tool_calling_agent,create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from retriever import retriever_tool
from langchain.schema.runnable import RunnableLambda
from search import search,url_searching
from scrapper import scrapper_tool
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

model = ChatHuggingFace(llm=llm)

prompt_template = prompt_template = """
You are an AI assistant helping users by searching for information and retrieving detailed answers. 

You can use these tools:
{tools}

TOOL NAMES:
{tool_names}

You must first use the `url_searching` tool to get a string of URLs.  
Then, you must use the `scrapper` tool to scrape content from the string of URLs you found.  

STRICT FORMAT:
Thought: [reason about next step]
Action: [tool_name]
Action Input: [input for the tool]
Output : [output obtained from the tool]

When you're ready to answer:
Final Answer: [final answer]
DO NOT answer unrelated questions.
DO NOT continue once the task is complete.

EXAMPLE:

Question: What are today's top news stories?
Thought: I need to find recent news articles.
Action: url_searching
Action Input: today's top news
Output : [list of URLs]

(After tool gives URLs)

Thought: I will now scrape content from the URLs to get detailed information.
Action: scrapper
Action Input: list of URLs
Output : [output from the scrapper]

(After tool gives content)

Thought: Now I can give the top news stories.
Final Answer: [output from the scrapper]

Scratchpad:
{agent_scratchpad}

Question: {input}
Your response:
"""

prompt = PromptTemplate.from_template(prompt_template)

llm_with_tools = RunnableLambda(lambda x, **kwargs: llm.invoke(x))

tools = [url_searching, scrapper_tool]

agent = create_react_agent(llm=llm_with_tools, tools=tools, prompt=prompt)

# agent = create_tool_calling_agent(llm_with_tools, tools, prompt = prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

history = ChatMessageHistory()

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id : history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

history.clear()

agent_with_history.invoke(
    {"input" : "tell me today's top news"},
    config = {"configurable" : {"session_id" : "<foo>"}},
)
