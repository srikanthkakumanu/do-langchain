from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

prompt = "What is smallest particle in the atomic structure?"

# Initialize a dictionary of models to compare
llms = {
    "OpenAI (gpt-5-nano)": ChatOpenAI(model="gpt-5-nano", temperature=0),
    "Groq (llama-3.1-8b-instant)": ChatGroq(
        model="llama-3.1-8b-instant", temperature=0
    ),
    "Google (gemini-2.5-flash)": ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0
    ),
}

# Invoke each model and print its response
for name, llm in llms.items():
    print(f"--- Calling {name} ---")
    try:
        response = llm.invoke(prompt)
        print(f"Response:\n{response.content}\n")
    except Exception as e:
        # Add error handling in case an API key is missing or another issue occurs
        print(f"Error calling {name}: {e}\n")
