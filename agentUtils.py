from langchain_openai.chat_models import  ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import  HumanMessage

class AgentUtils:
    @staticmethod
    def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
        """Create a function-calling agent and add it to the graph."""
        system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
        system_prompt += " Do not ask for clarification."
        system_prompt += " Your other team members (and other teams) will collaborate with you with their own specialties."
        system_prompt += " You are chosen for a reason! You are one of the following team members: {team_members}."
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        return executor

    @staticmethod
    def agent_node(state, agent, name):
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}
    
    @staticmethod
    def create_team_supervisor(llm: ChatOpenAI, system_prompt: str, members: list):
        """An LLM-based router."""
        options = ["FINISH"] + members
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ],
                    },
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next?"
                    " Or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(options), team_members=", ".join(members))
        return (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )
