from dotenv import load_dotenv
import os
load_dotenv()

def main():
    print("Hello from langchain-course!")
    print(os.getenv("AZURE_OPENAI_ENDPOINT"))


if __name__ == "__main__":
    main()
