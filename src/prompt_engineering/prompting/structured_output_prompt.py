"""
Structured output & parsing via native tool-calling using LangChain v1's
unified chat model interface (init_chat_model).

model.with_structured_output(Schema) has the provider enforce a Pydantic
schema at generation time, instead of just asking for JSON in the prompt
text (see output_formatting_prompt.py for that weaker, request-only version).
The response arrives as a validated, typed object -- no manual json.loads,
no "the model added a preamble before the JSON" parsing failures.
"""

from pydantic import BaseModel, Field

from utils.llm_utils import get_model, load_environment_variables

QUERY = "Give me a recipe for pancakes."


class Recipe(BaseModel):
    """A recipe extracted from the model's response."""

    name: str = Field(description="Recipe name")
    ingredients: list[str] = Field(description="Main ingredients")
    prep_time_minutes: int = Field(description="Preparation time in minutes")


def print_recipe(recipe: Recipe, model_name: str):
    print(f"\n\n--- Response from {model_name}: ---")
    print(f"Name: {recipe.name}")
    print(f"Ingredients: {recipe.ingredients}")
    print(f"Prep time (minutes): {recipe.prep_time_minutes}")


def _ask(model_name: str) -> Recipe:
    llm = get_model(model_name)
    structured_llm = llm.with_structured_output(Recipe)
    return structured_llm.invoke(QUERY)


def llama():
    return _ask("llama")


def gemini():
    return _ask("gemini")


def openai():
    return _ask("openai")


def claude():
    return _ask("claude")


def main():
    """Main function to invoke models with a schema-enforced query and print the parsed recipe."""

    load_environment_variables()

    for model_name, fn in (("llama", llama), ("gemini", gemini), ("openai", openai)):
        recipe = fn()
        print_recipe(recipe, model_name)


if __name__ == "__main__":
    main()
