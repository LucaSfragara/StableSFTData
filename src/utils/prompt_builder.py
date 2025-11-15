def build_prompt(system_prompt: str, question: str, answer: str | None = None) -> str:
    
    # canonical Tulu-style format
    prompt = (
        "<|user|>\n"
        + system_prompt.strip() + "\n\n"
        + question.strip() + "\n"
        "<|assistant|>\n"
    )
    if answer is not None:
        prompt += answer.strip()
    return prompt