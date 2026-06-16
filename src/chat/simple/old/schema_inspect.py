"""
LangChain - Core Concepts: Schema Inspection - Old Way
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = "Summarize the following text: {text}"


def inspect_schema_openai():
    """
    Inspect schema using OpenAI
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    print(f"Input Schema: {chain.input_schema.model_json_schema()}")
    print(f"Output Schema: {chain.output_schema.model_json_schema()}")


def inspect_schema_google():
    """
    Inspect schema using Google Gemini
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    print(f"Input Schema: {chain.input_schema.model_json_schema()}")
    print(f"Output Schema: {chain.output_schema.model_json_schema()}")


def inspect_schema_groq():
    """
    Inspect schema using Groq
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    print(f"Input Schema: {chain.input_schema.model_json_schema()}")
    print(f"Output Schema: {chain.output_schema.model_json_schema()}")


def main():
    # print("--- Calling OpenAI ---")
    # inspect_schema_openai()
    print("\n--- Calling Google Gemini ---")
    inspect_schema_google()
    print("\n--- Calling Groq ---")
    inspect_schema_groq()


if __name__ == "__main__":
    main()
