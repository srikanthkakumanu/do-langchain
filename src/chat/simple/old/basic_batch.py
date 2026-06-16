"""
LangChain Core Concepts - LCEL and Runnables (chain) and batch execution - Old Way
"""

from dotenv import load_dotenv, main
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = "You are a helpful assistant. Answer in one sentence: {question}"

prompt_template = "Translate to french: {text}"


def basic_batch_execution_openai():
    """Demonstrate basic batch execution for multiple inputs with OpenAI"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} -> Output: {text[1]}")


def basic_batch_execution_google():
    """Demonstrate basic batch execution for multiple inputs with Google Gemini"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} -> Output: {text[1]}")


def basic_batch_execution_groq():
    """Demonstrate basic batch execution for multiple inputs with Groq"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} -> Output: {text[1]}")


def main():
    # print("--- Calling OpenAI ---")
    # basic_batch_execution_openai()
    print("\n--- Calling Google Gemini ---")
    basic_batch_execution_google()
    print("\n--- Calling Groq ---")
    basic_batch_execution_groq()


if __name__ == "__main__":
    main()
