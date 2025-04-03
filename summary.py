import sys
from openai import OpenAI

def summarize_file(file_path: str, model_name) -> list:
    """
    Reads the given file, sends its content to the OpenAI API for summarization,
    and returns a list containing [summary_text, total_token_usage].
    """

    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "user", "content": f"Summarize the following content not to long (four sentences max) be technically precise:\n\n{file_content}"}
        ],
        temperature=0.7,
    )

    summary = response.choices[0].message.content
    total_tokens = response.usage.total_tokens

    return [summary, total_tokens]