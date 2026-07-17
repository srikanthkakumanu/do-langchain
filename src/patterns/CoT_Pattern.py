"""
Chain-of-Thought (CoT) pattern.

The model is asked to reason step by step, in an explicit ordered sequence,
before giving a final answer -- instead of jumping straight to a conclusion.

Unlike ReAct, there is no tool loop here: it's a single model call, driven
entirely by the prompt asking for step-by-step reasoning.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

cot_prompt = ChatPromptTemplate.from_template(
    """Solve the problem below. Reason step by step, then give the
answer on its own line as "Final Answer: <answer>".

Question: {question}
"""
)

# provider:model strings understood by init_chat_model.
# Each provider needs its own API key in .env (GOOGLE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GROQ_API_KEY).
models = {
    "gemini": "google_genai:gemini-2.5-flash",
    "openai": "openai:gpt-5-nano",
    "claude": "anthropic:claude-opus-4-8",
    "groq": "groq:llama-3.3-70b-versatile",
}


def build_chain(model: str):
    return cot_prompt | init_chat_model(model) | StrOutputParser()


def run(question: str, model_key: str = "gemini"):
    chain = build_chain(models[model_key])
    result = chain.invoke({"question": question})

    reasoning, _, answer = result.partition("Final Answer:")
    print(f"Question: {question}\n")
    print(f"Reasoning:{reasoning.strip()}\n")
    print(f"Final Answer: {answer.strip()}")


if __name__ == "__main__":
    question = (
        "A store has 12 apples. It sells 5 and receives a delivery of 8 more. "
        "How many apples does it have now?"
    )

    for model_key in models:
        print(f"\n=== {model_key} ({models[model_key]}) ===\n")
        run(question, model_key)
