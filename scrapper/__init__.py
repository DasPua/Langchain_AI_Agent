from scrapper.scrapper import Scrapper_func
from langchain_community.tools import Tool

scrapper_tool = Tool(
    name="web_scrapper",
    description="Scrapes and summarizes web content from a string of URLs. Input should be a string of URLs (comma-separated).",
    func=Scrapper_func
)