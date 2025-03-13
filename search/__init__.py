from .travily import TavilySearchResults
from langchain_community.tools import Tool

search = TavilySearchResults()

url_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_raw_content=False
)

def collect_urls(text):
    collected_urls = ""

    for output in url_search.invoke(text):
        collected_urls+=(output['url']) +","
    collected_urls = collected_urls[:-1]
    return collected_urls

url_searching = Tool(
    name="url_searching",
    description="Searches for URLs related to a given topic and returns a string of URLs. Takes string as an input",
    func = collect_urls
)

