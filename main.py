from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    return len(text)


def main():
    print("Loading environment variables from .env file...")
    tools = [get_text_length]
    
    # Create the LLM with tool calling support
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    
    # Create the agent using LangGraph (modern tool calling approach)
    agent_executor = create_react_agent(llm, tools)
    
    # Invoke the agent
    response = agent_executor.invoke(
        {"messages": [HumanMessage(content="What is the length of the text 'Hello, world!'?")]}
    )
    
    print("\nFinal response:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
