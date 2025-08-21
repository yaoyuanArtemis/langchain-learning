import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max=2)

respond = search.invoke("苹果2025发布会")
print(respond)