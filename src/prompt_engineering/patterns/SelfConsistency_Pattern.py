"""
Self-Consistency pattern.

Instead of trusting a single Chain-of-Thought pass, sample several
*independent* reasoning paths for the same question (at temperature > 0)
and take a majority vote over their final answers. Wrong reasoning tends to
fail in different, uncorrelated ways across samples, while correct reasoning
tends to converge on the same answer -- so the most common answer is more
likely to be right than any single path.
"""

from collections import Counter

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
    sampling_model = init_chat_model(model, temperature=0.7)
    return cot_prompt | sampling_model | StrOutputParser()


def sample_answers(question: str, model: str, n: int = 5) -> list[str]:
    chain = build_chain(model)
    answers = []
    for _ in range(n):
        result = chain.invoke({"question": question})
        _, _, answer = result.partition("Final Answer:")
        answers.append(answer.strip())
    return answers


def run(question: str, model_key: str = "gemini", n: int = 5):
    answers = sample_answers(question, models[model_key], n=n)
    most_common, votes = Counter(answers).most_common(1)[0]

    print(f"Question: {question}\n")
    print(f"Samples: {answers}")
    print(f"Final Answer: {most_common} ({votes}/{n} votes)")


if __name__ == "__main__":
    question = (
        "A store has 12 apples. It sells 5 and receives a delivery of 8 more. "
        "How many apples does it have now?"
    )

    for model_key in models:
        print(f"\n=== {model_key} ({models[model_key]}) ===\n")
        run(question, model_key)
