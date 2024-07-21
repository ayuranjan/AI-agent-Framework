from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

class LangaugeAgentTools:
  """
  A class containing static methods for language tasks.
  """
  @staticmethod
  @tool
  def tavilySearch():
    """
    A tool for searching the web using the Tavily search engine.
    """
    return TavilySearchResults(max_results=5) 
  

  