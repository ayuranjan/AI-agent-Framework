from typing_extensions import TypedDict
from typing import List, Annotated
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint import MemorySaver
class VisionLanguageTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    
    team_members: List[str]
    
    next: str


class MultiAgentGraphManager:
  """
  A class to manage a state graph for multiple agents.
  """

  def __init__(self, vision_language_node, video_node, language_node, supervisor, state_type=VisionLanguageTeamState ):
    """
    Initializes the state graph with the specified state type.

    Args:
      state_type (class, optional): The type of state object used in the graph. Defaults to VisionLanguageTeamState.
    """
    self.agent_graph = StateGraph(state_type)
    self.checkpointer = MemorySaver()

    # Pre-define nodes (can be modified if needed)
    self.agent_graph.add_node("Vision_Language_Agent", vision_language_node)
    self.agent_graph.add_node("Video_Agent", video_node)
    self.agent_graph.add_node("Language_Agent", language_node)
    self.agent_graph.add_node("supervisor", supervisor)

    # Pre-define edges (can be modified if needed)
    self.agent_graph.add_edge("Language_Agent", "supervisor")
    self.agent_graph.add_edge("Vision_Language_Agent", "supervisor")
    self.agent_graph.add_edge("Video_Agent", "supervisor")

    self.agent_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"Vision_Language_Agent": "Vision_Language_Agent",
         "Language_Agent": "Language_Agent",
         "Video_Agent": "Video_Agent",
         "FINISH": END},
    )

    self.agent_graph.add_edge(START, "supervisor")
    self.research_chain = self.agent_graph.compile(checkpointer=self.checkpointer)