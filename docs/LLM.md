# LLM Providers and Cheapest Models

Here is the comprehensive list of LLM providers and models that are cheapest and simplest to use and some of them are free.

| Company Name | Model                                     | Model Provider | Cost Filter   | Rate / Free Status (May 2026)                    |
| ------------ | ----------------------------------------- | -------------- | ------------- | ------------------------------------------------ |
| Google       | gemini-2.5-flash-lite                     | google_genai   | Free          | Free Tier (15 RPM) / $0.10 in / $0.40 out (Paid) |
| Google       | gemini-2.5-flash                          | google_genai   | Free          | Free Tier (15 RPM) / $0.10 in / $0.40 out (Paid) |
| Google       | gemini-3.1-flash                          | google_genai   | Free          | Free Tier (15 RPM) / $0.10 in / $0.40 out (Paid) |
| Groq         | llama-3.1-8b-instant                      | groq           | Free          | Free Tier (Up to 14,400 requests/day)            |
| Groq         | llama-3.3-70b-versatile                   | groq           | Free          | Free Tier (Lower daily limits than 8B)           |
| Groq         | meta-llama/llama-4-scout-17b-16e-instruct | groq           | Free          | Free Tier (High-efficiency reasoning model)      |
| Groq         | mixtral-8x7b-32768                        | groq           | Free          | Free Tier (Great for balanced performance)       |
| Groq         | qwen/qwen3-32b                            | groq           | Free          | Free Tier (Rate limits apply)                    |
| Mistral      | ministral-3b-latest                       | mistralai      | Cheapest Paid | $0.06 Input / $0.06 Output                       |
| OpenAI       | gpt-5-nano                                | openai         | Low Cost Paid | $0.05 Input / $0.40 Output                       |
| Mistral      | mistral-small-latest                      | mistralai      | Low Cost Paid | ~$0.10 Input / $0.30 Output                      |
| Mistral      | mistral-large-latest                      | mistralai      | Costly Paid   | ~$2.00 Input / $6.00 Output                      |

## Text Embedding Models

Here is a comparison of free and cheapest text embedding models across major providers. Note that **Anthropic does not offer a native embedding model** — they officially recommend [Voyage AI](https://www.voyageai.com/) as their preferred third-party embeddings partner.

| Company Name | Model                                  | Model Provider | Cost Filter   | Vector Dimensions                           | Rate / Free Status (May 2026)                                               |
| ------------ | -------------------------------------- | -------------- | ------------- | ------------------------------------------- | --------------------------------------------------------------------------- |
| Google       | gemini-embedding-001                   | google_genai   | Free          | 3072 (configurable: 768/1536/3072)          | Free Tier (rate-limited) / $0.15 per 1M tokens (Paid)                       |
| HuggingFace  | sentence-transformers/all-MiniLM-L6-v2 | huggingface    | Free          | 384                                         | Free (Inference API, rate-limited) / Free to self-host                      |
| HuggingFace  | BAAI/bge-small-en-v1.5                 | huggingface    | Free          | 384                                         | Free (Inference API, rate-limited) / Free to self-host                      |
| Mistral      | mistral-embed                          | mistralai      | Cheapest Paid | 1024                                        | ~$0.01 per 1M tokens                                                        |
| OpenAI       | text-embedding-3-small                 | openai         | Cheapest Paid | 1536 (configurable, shorter dims supported) | $0.02 per 1M tokens                                                         |
| OpenAI       | text-embedding-3-large                 | openai         | Low Cost Paid | 3072 (configurable, shorter dims supported) | $0.13 per 1M tokens                                                         |
| Groq         | —                                      | groq           | Not Available | N/A                                         | Groq does not currently provide an embeddings endpoint                      |
| Anthropic    | —                                      | anthropic      | Not Available | N/A (Voyage AI: 512–1024 typical)           | No native embedding model — recommends Voyage AI (voyage-3-lite / voyage-3) |

## Multimodal Models (Video, Image, Audio)

Here is a comparison of free and cheapest multimodal models — covering image understanding/generation, audio (speech-to-text), and video input — across major providers. Note that **Anthropic's Claude models support image input natively but have no audio, video, or generative (image/audio output) models** — Anthropic does not offer a dedicated multimodal generation product.

| Company Name | Model                                     | Model Provider | Modality                     | Cost Filter   | Rate / Free Status (May 2026)                              |
| ------------ | ----------------------------------------- | -------------- | ---------------------------- | ------------- | ---------------------------------------------------------- |
| Google       | gemini-2.5-flash-lite                     | google_genai   | Image + Audio + Video + Text | Free          | Free Tier (15 RPM) / $0.10 in / $0.40 out (Paid)           |
| Google       | gemini-2.5-flash                          | google_genai   | Image + Audio + Video + Text | Free          | Free Tier (15 RPM) / $0.10 in / $0.40 out (Paid)           |
| Groq         | meta-llama/llama-4-scout-17b-16e-instruct | groq           | Image + Text                 | Free          | Free Tier (High-efficiency vision-capable model)           |
| Groq         | whisper-large-v3-turbo                    | groq           | Audio (Speech-to-Text)       | Free          | Free Tier (Fast transcription, rate limits apply)          |
| HuggingFace  | llava-hf/llava-1.5-7b-hf                  | huggingface    | Image + Text                 | Free          | Free (Inference API, rate-limited) / Free to self-host     |
| HuggingFace  | openai/whisper-large-v3                   | huggingface    | Audio (Speech-to-Text)       | Free          | Free (Inference API, rate-limited) / Free to self-host     |
| HuggingFace  | stabilityai/stable-diffusion-3.5-large    | huggingface    | Image Generation             | Free          | Free (Inference API, rate-limited) / Free to self-host     |
| Mistral      | pixtral-12b-latest                        | mistralai      | Image + Text                 | Cheapest Paid | ~$0.15 Input / $0.15 Output per 1M tokens                  |
| OpenAI       | gpt-4o-mini                               | openai         | Image + Text                 | Low Cost Paid | $0.15 Input / $0.60 Output per 1M tokens                   |
| OpenAI       | whisper-1                                 | openai         | Audio (Speech-to-Text)       | Cheapest Paid | $0.006 per minute                                          |
| OpenAI       | gpt-image-1-mini                          | openai         | Image Generation             | Cheapest Paid | ~$0.02 per image (low quality setting)                     |
| Anthropic    | claude-haiku-4-5                          | anthropic      | Image + Text                 | Low Cost Paid | $1.00 Input / $5.00 Output per 1M tokens (vision included) |
