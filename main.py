from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Elon musk is the CEO of SpaceX and Tesla.
    He was born in South Africa and later moved to the United States.
    He is known for his work in the fields of space exploration, electric vehicles, and renewable energy.
    """
    summary_template = """
    Summarize the following information in a concise manner:
    {information}
    1. provide short summary
    2. two interesting facts about the person
    """

    summry_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    )

    chain = summry_prompt_template | llm
    response = chain.invoke({"information": information})
    print("Response from Azure OpenAI:")
    print(response.content)


if __name__ == "__main__":
    main()
