from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen3:4b"

# Default sampling temperature. main.py leaves this at 0.0 (deterministic);
# experiment.py raises it so repeated trials actually vary.
TEMPERATURE = 0.0


def ask_qwen(system_prompt: str, user_prompt: str, temperature: float = None) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE if temperature is None else temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content
