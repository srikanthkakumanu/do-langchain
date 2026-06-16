"""
Old Way
Demonstrates a first custom chain using LCEL and Runnables using OpenAI, Google Gemini, and Groq.
Create a chain that:
1. Takes a product name and target audience
2. Generates a marketing tagline
3. Returns just the tagline as a string
Test with: product="AI Course", audience="developers"
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = (
    "Create a marketing tagline for a product named {product} targeting {audience}."
)


def first_chain_openai():
    """A first custom chain using LCEL and Runnables using OpenAI."""

    # Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Response: {result}")

    return chain


def first_chain_google():
    """A first custom chain using LCEL and Runnables using Google Gemini."""

    # Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Response: {result}")

    return chain


def first_chain_groq():
    """A first custom chain using LCEL and Runnables using Groq."""

    # Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Response: {result}")

    return chain


def main():
    # print("--- Calling OpenAI ---")
    # first_chain_openai()

    print("--- Calling Google Gemini ---")
    first_chain_google()

    print("--- Calling Groq ---")
    first_chain_groq()


if __name__ == "__main__":
    main()
