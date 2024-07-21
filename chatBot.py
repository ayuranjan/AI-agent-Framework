from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import  HumanMessage
from multiAgentGraphManager import MultiAgentGraphManager
from  visionLanguageAgentTools import VisionLanguageAgentTools
from languageAgentTools import LangaugeAgentTools
from agentUtils import AgentUtils
from agentsPrompts import AgentPrompts
import functools
from langchain_openai.chat_models import  ChatOpenAI

def without_steaming(research_chain, prompt, config):
    previous_message = None
    final_output = None

    for s in research_chain.stream({"messages": [prompt]}, config):
        if "__end__" not in s:
            if 'supervisor' in s:
                agent_name = s['supervisor']['next']
                if agent_name != 'FINISH':
                    st.write(f"Supervisor calling \"{agent_name}\"")
                    print(f"Supervisor calling \"{agent_name}\"")
            elif 'Language_Agent' in s:
                message = s['Language_Agent']['messages'][0].content
                st.write(f"Language Agent: {message}")
                print(f"Language Agent: {message}")
            elif 'Vision_Language_Agent' in s:
                message = s['Vision_Language_Agent']['messages'][0].content
                st.write(f"Vision Language Agent: {message}")
                print(f"Vision Language Agent: {message}")
            elif 'Video_Agent' in s:
                message = s['Video_Agent']['messages'][0].content
                st.write(f"Video Agent: {message}")
                print(f"Video Agent: {message}")

            # Keep track of the final output message
            if 'supervisor' in s and s['supervisor']['next'] == 'FINISH':
                final_output = previous_message
            previous_message = s


        if final_output:
            agent_name = list(final_output.keys())[0]
            final_message = final_output[agent_name]['messages'][0].content
            # st.markdown(f"Final Output: {final_message}")
            print(f"Final Output: {final_message}")
            
            
            
def chatBot(research_chain):
    config = {"configurable": {"thread_id": 42}}
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            without_steaming(research_chain , prompt= prompt, config=config)
            
            

    


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
    chatBot(manager.research_chain)