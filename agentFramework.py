from dotenv import load_dotenv
from langchain_openai.chat_models import  ChatOpenAI
from langchain_core.messages import  HumanMessage
from multiAgentGraphManager import MultiAgentGraphManager
from  visionLanguageAgentTools import VisionLanguageAgentTools
from languageAgentTools import LangaugeAgentTools
from agentUtils import AgentUtils
from agentsPrompts import AgentPrompts
import functools
import matplotlib.pyplot as plt


def interactive_prompt( research_chain  ):
    config = {"configurable": {"thread_id": 42}}
    
    print("Enter your question (type 'exit' to quit):")
    
    while True:
        user_input = input(">> ")
        
        if user_input.lower() == 'exit':
            print("Exiting the interactive prompt.")
            break
        
        # Create a message from user input
        input_message = HumanMessage(content=user_input)
        
        # Stream the response from the Langraph agent framework
        for s in research_chain.stream({"messages": [input_message]}, config):
            if "__end__" not in s:
                print(s)
                print("---")
              
def toolsForVisionLanguageAgent():
    return [VisionLanguageAgentTools.VisualQuestionAnswer, VisionLanguageAgentTools.clickNewImage]
    

def toolsForVideoAgent():
    return []

def toolsForLanguageAgent():
    return [LangaugeAgentTools.tavilySearch]



if __name__ == "__main__":
    load_dotenv() 
    llm = ChatOpenAI(model="gpt-4-1106-preview")

    vision_language_agent = AgentUtils.create_agent( llm, toolsForVisionLanguageAgent(),AgentPrompts.vision_language_agent_prompt)
    vision_language_node = functools.partial(AgentUtils.agent_node, agent=vision_language_agent, name="Vision_Language_Agent")

    video_agent = AgentUtils.create_agent( llm, toolsForVideoAgent(), AgentPrompts.video_agent_prompt)
    video_node = functools.partial(AgentUtils.agent_node, agent=video_agent, name="Video_Agent")

    language_agent = AgentUtils.create_agent( llm, toolsForLanguageAgent(), AgentPrompts.language_agent_prompt)
    language_node = functools.partial(AgentUtils.agent_node, agent=language_agent, name="Language_Agent")

    supervisor_agent = AgentUtils.create_team_supervisor( llm, AgentPrompts.supervisor_agent_prompt,["Language_Agent","Vision_Language_Agent", "Video_Agent"])

    manager = MultiAgentGraphManager(vision_language_node, video_node, language_node, supervisor_agent)

    interactive_prompt(manager.research_chain) 



