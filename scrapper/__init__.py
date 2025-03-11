from scrapper.scrapper import Scrapper_func
from langchain_community.tools import Tool

scrapper = Tool(
    name="web_scrapper",
    description="Scrapes and summarizes web content from a list of URLs. Input should be a list of URLs.",
    func=Scrapper_func
)