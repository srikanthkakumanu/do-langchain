"""
Tree-of-Thought (ToT) pattern.

Generalizes Chain-of-Thought from a single reasoning path into a search over
several candidate paths: at each step the model proposes a few possible next
thoughts, an evaluator chain scores each candidate, and a breadth-first
search keeps only the best-scoring branches -- discarding the rest instead of
committing to the first idea.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

propose_prompt = ChatPromptTemplate.from_template(
    """Problem: {problem}
So far: {state}

Propose {k} different, distinct next steps toward solving this problem.
List them one per line, no numbering.
"""
)

evaluate_prompt = ChatPromptTemplate.from_template(
    """Problem: {problem}
Candidate reasoning so far: {state}

Rate how promising this path is toward a correct final answer,
from 0 (dead end) to 10 (clearly on track). Reply with just the number.
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


def build_chains(model: str):
    llm = init_chat_model(model)
    propose_chain = propose_prompt | llm | StrOutputParser()
    evaluate_chain = evaluate_prompt | llm | StrOutputParser()
    return propose_chain, evaluate_chain


def score(evaluate_chain, problem: str, state: str) -> float:
    raw = evaluate_chain.invoke({"problem": problem, "state": state})
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else 0.0
    except ValueError:
        return 0.0


def tree_of_thought(
    problem: str, model: str, k: int = 3, breadth: int = 2, depth: int = 3
) -> str:
    propose_chain, evaluate_chain = build_chains(model)
    states = [""]  # each state is the reasoning accumulated so far

    for _ in range(depth):
        candidates = []
        for state in states:
            proposals = propose_chain.invoke(
                {"problem": problem, "state": state, "k": k}
            )
            for line in proposals.splitlines():
                if line.strip():
                    candidates.append(f"{state}\n{line}".strip())

        scored = [(score(evaluate_chain, problem, c), c) for c in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        states = [state for _, state in scored[:breadth]]  # keep the best branches

    return states[0]  # best final path


def run(problem: str, model_key: str = "gemini"):
    best_path = tree_of_thought(problem, models[model_key])
    print(f"Problem: {problem}\n")
    print(f"Best reasoning path:\n{best_path}")


if __name__ == "__main__":
    problem = (
        "Using the numbers 4, 7, 8, and 8 exactly once each, with +, -, *, /, "
        "make the number 24."
    )

    for model_key in models:
        print(f"\n=== {model_key} ({models[model_key]}) ===\n")
        run(problem, model_key)
