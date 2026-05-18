from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool


load_dotenv()

# Tool Creation


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number."""
    return x**0.5


@tool("add")
def add(x: int, y: int) -> int:
    """Calculate the sum of two numbers."""
    return x + y


@tool("multiply", description="Calculate the product of two numbers.")
def multiply(x: int, y: int) -> int:
    return x * y


# print(square_root.invoke({"x": 16}))
# print(add.invoke({"x": 5, "y": 7}))
# print(multiply.invoke({"x": 3, "y": 4}))

# Adding Tools to an Agent


def createAgent():
    agent = create_agent(
        model="google_genai:gemini-2.5-flash",
        tools=[square_root, add, multiply],
        system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root, sum, or product of numbers.",
    )
    return agent


question = HumanMessage(content="What is the square root of 467?")

response = createAgent().invoke({"messages": [question]})

# Iterate through the message history to inspect tool calls
for message in response["messages"]:
    if getattr(message, "tool_calls", None):
        print(
            f"Tool calls found in {message.__class__.__name__}:\n{message.tool_calls}\n"
        )

print(f"Final Answer: {response['messages'][-1].content}")
