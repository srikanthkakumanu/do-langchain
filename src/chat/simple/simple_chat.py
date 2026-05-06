from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from pprint import pprint

load_dotenv()

prompt = "What is the capital of France?"

llms = {
    "llama-3.1-8b-instant": init_chat_model(
        model="llama-3.1-8b-instant", model_provider="groq", temperature=0
    ),
    "gemini-2.5-flash": ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0),
    # "OpenAI (gpt-5-nano)": init_chat_model(model="gpt-5-nano", temperature=0),
}

for model_name, llm in llms.items():
    pprint(f"--- Calling {model_name} ---")
    response = llm.invoke(prompt)
    # response, content, to_json, to_dict
    pprint(f"Response from {model_name}: {response.response_metadata}")
