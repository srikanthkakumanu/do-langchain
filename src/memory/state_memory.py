from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

load_dotenv()

agent = create_agent("groq:llama-3.1-8b-instant", checkpointer=InMemorySaver())

question = HumanMessage(
    content="Hello My Name is John and What is the oldest galaxy in the universe?"
)
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke({"messages": [question]}, config=config)

print("First Call: \n", response)

question = HumanMessage(content="What is the oldest star in the universe?")
response = agent.invoke({"messages": [question]}, config=config)

print("\n Second Call: \n", response)
