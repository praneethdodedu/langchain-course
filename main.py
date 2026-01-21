from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_classic import hub
from langchain_core.tools.render import render_text_description

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    return len(text)


def main():
    print("Loading environment variables from .env file...")
    # print(get_text_length("Hello, world!"))  # Example usage of the utility function
    tools = [get_text_length]
    template = """
    Answer the following question using the provided tools.
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original question

    begin!
    Question: {input}
    Thought: 
    """
    prompt = PromptTemplate.from_template(template).partial(
        tool_names=", ".join([tool.name for tool in tools]), tools=render_text_description(tools)
    )
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    agent = {"input": lambda x: x["input"]} | prompt | llm 
    response = agent.invoke(
        {"input": "What is the length of the text 'Hello, world!'?"}
    )
    print(response)


if __name__ == "__main__":
    main()
