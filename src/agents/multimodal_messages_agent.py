import base64
import io
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

MOON_IMAGE_PATH = (
    Path(__file__).resolve().parent.parent
    / "context_engineering"
    / "resources"
    / "moon.png"
)


SCI_FI_WRITER_PROMPT = "You are a science fiction writer, create a capital city at the users request."


def build_agent(model: str, system_prompt: str | None = None):
    return create_agent(model=model, system_prompt=system_prompt)


def text_input():
    agent = build_agent("google_genai:gemini-2.5-flash", system_prompt=SCI_FI_WRITER_PROMPT)

    question = HumanMessage(
        content=[{"type": "text", "text": "What is the capital of The Moon?"}]
    )

    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


def image_input(image_path: Path = MOON_IMAGE_PATH):
    agent = build_agent("google_genai:gemini-2.5-flash", system_prompt=SCI_FI_WRITER_PROMPT)

    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    multimodal_question = HumanMessage(
        content=[
            {"type": "text", "text": "Tell me about this capital"},
            {"type": "image", "base64": img_b64, "mime_type": "image/png"},
        ]
    )

    response = agent.invoke({"messages": [multimodal_question]})
    print(response["messages"][-1].content)


def record_audio_base64(duration: int = 5, sample_rate: int = 44100) -> str:
    import sounddevice as sd
    from scipy.io.wavfile import write

    print(f"Recording for {duration}s...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Done.")

    buf = io.BytesIO()
    write(buf, sample_rate, audio)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def audio_input():
    aud_b64 = record_audio_base64()

    agent = build_agent("gpt-4o-audio-preview")

    multimodal_question = HumanMessage(
        content=[
            {"type": "text", "text": "Tell me about this audio file"},
            {"type": "audio", "base64": aud_b64, "mime_type": "audio/wav"},
        ]
    )

    response = agent.invoke({"messages": [multimodal_question]})
    print(response["messages"][-1].content)


def main():
    print("\n=== Text input ===\n")
    text_input()

    print("\n=== Image input ===\n")
    image_input()

    print("\n=== Audio input ===\n")
    audio_input()


if __name__ == "__main__":
    main()
