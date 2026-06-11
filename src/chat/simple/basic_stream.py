"""
LangChain Core Concepts - LCEL and Runnables (chain) and streaming
"""

from dotenv import load_dotenv, main
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = "Write a haiku about: {topic}"


def basic_stream_openai():
    """Demonstrate basic streaming for real-time output with OpenAI"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.7,
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def basic_stream_google():
    """Demonstrate basic streaming for real-time output with Google Gemini"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def basic_stream_groq():
    """Demonstrate basic streaming for real-time output with Groq"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def main():
    # print("--- Calling OpenAI ---")
    # basic_stream_openai()
    print("\n--- Calling Google Gemini ---")
    basic_stream_google()
    print("\n--- Calling Groq ---")
    basic_stream_groq()


if __name__ == "__main__":
    main()
