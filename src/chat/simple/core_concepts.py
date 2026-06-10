"""
LangChain Core Concepts - LCEL and Runnables
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = "You are a helpful assistant. Answer in one sentence: {question}"


def basic_chain_openai():
    """Demonstrates basic chain using LCEL and Runnables with OpenAI."""

    # Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain


def basic_chain_google():
    """Demonstrates basic chain using LCEL and Runnables with Google Gemini."""

    # Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain


def basic_chain_groq():
    """Demonstrates basic chain using LCEL and Runnables with Groq."""

    # Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain


def main():
    # print("--- Calling OpenAI ---")
    # basic_chain_openai()
    print("\n--- Calling Google Gemini ---")
    basic_chain_google()
    print("\n--- Calling Groq ---")
    basic_chain_groq()


if __name__ == "__main__":
    main()
