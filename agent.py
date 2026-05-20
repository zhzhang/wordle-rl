import re
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from game import pick_word, score_guess

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MAX_ATTEMPTS = 6


def _format_feedback(guess: str, scores: list[str]) -> str:
    return " ".join(f"{g.upper()}:{s}" for g, s in zip(guess, scores))


def _extract_guess(text: str) -> str | None:
    m = re.search(r"(?i)guess[:\s]+([a-zA-Z]{5})\b", text)
    if m:
        return m.group(1).lower()
    words = re.findall(r"\b[a-zA-Z]{5}\b", text)
    return words[-1].lower() if words else None


def run_agent(target: str | None = None) -> bool:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    word = target if target is not None else pick_word()

    system_prompt = textwrap.dedent("""\
        You are playing Wordle. Rules:
        - Guess a 5-letter English word each turn.
        - For each letter you receive feedback:
            correct  - right letter, right position
            present  - right letter, wrong position
            absent   - letter not in the word
        - Reply with: GUESS: <word>
    """)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Make your first guess."},
    ]

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        full_response = tokenizer.decode(new_tokens, skip_special_tokens=False)

        think_match = re.search(r"<think>.*?</think>(.*)", full_response, re.DOTALL)
        answer_text = (think_match.group(1) if think_match else full_response).strip()

        guess = _extract_guess(answer_text)
        if guess is None:
            print(f"Attempt {attempt_num}: could not parse a 5-letter guess. Run failed.")
            return False

        try:
            scores = score_guess(word, guess)
        except ValueError as exc:
            print(f"Attempt {attempt_num}: invalid guess '{guess}' ({exc}). Run failed.")
            return False

        feedback = _format_feedback(guess, scores)
        print(f"Attempt {attempt_num}: {guess.upper()} -> {feedback}")

        if all(s == "correct" for s in scores):
            print(f"Solved in {attempt_num} attempt(s). Word: {word.upper()}")
            return True

        messages.append({"role": "assistant", "content": answer_text})
        messages.append({
            "role": "user",
            "content": f"Feedback: {feedback}\nMake your next guess.",
        })

    print(f"Out of attempts. Word: {word.upper()}")
    return False
