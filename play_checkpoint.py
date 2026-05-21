import argparse
import re
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from game import pick_word, score_guess

MAX_ATTEMPTS = 6

SYSTEM_PROMPT = textwrap.dedent("""\
    You are playing Wordle. Rules:
    - Guess a 5-letter English word each turn.
    - For each letter you receive feedback:
        correct  - right letter, right position
        present  - right letter, wrong position
        absent   - letter not in the word
    - Reply with: GUESS: <word>
""")


def _extract_guess(text: str) -> str | None:
    match = re.search(r"(?i)guess[:\s]+([a-zA-Z]{5})\b", text)
    if match:
        return match.group(1).lower()
    words = re.findall(r"\b[a-zA-Z]{5}\b", text)
    return words[-1].lower() if words else None


def _format_feedback(guess: str, scores: list[str]) -> str:
    return " ".join(f"{g.upper()}:{s}" for g, s in zip(guess, scores))


def _split_reasoning_and_answer(response: str) -> tuple[str | None, str]:
    match = re.search(r"<think>(.*?)</think>(.*)", response, flags=re.DOTALL)
    if not match:
        return None, response.strip()
    reasoning = match.group(1).strip()
    answer = match.group(2).strip()
    return reasoning, answer


def _has_model_weights(checkpoint_dir: Path) -> bool:
    patterns = [
        "*.safetensors",
        "pytorch_model*.bin",
        "consolidated*.pth",
        "model*.safetensors.index.json",
        "pytorch_model*.bin.index.json",
    ]
    return any(any(checkpoint_dir.glob(pattern)) for pattern in patterns)


def _discover_latest_checkpoint(root: Path) -> Path:
    candidates: list[Path] = []
    for config_path in root.rglob("config.json"):
        checkpoint_dir = config_path.parent
        if _has_model_weights(checkpoint_dir):
            candidates.append(checkpoint_dir)

    if not candidates:
        raise FileNotFoundError(
            "Could not find a local checkpoint directory with config.json and weights. "
            "Pass --checkpoint-path explicitly if your files are elsewhere."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_checkpoint(explicit_path: str | None, search_root: str) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint path is not a directory: {path}")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Checkpoint directory missing config.json: {path}")
        if not _has_model_weights(path):
            raise FileNotFoundError(f"Checkpoint directory has no recognizable model weights: {path}")
        return path

    return _discover_latest_checkpoint(Path(search_root).expanduser().resolve())


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the newest checkpoint and play one Wordle game."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint directory to load.",
    )
    parser.add_argument(
        "--search-root",
        type=str,
        default=".",
        help="Root directory to search for the latest checkpoint when --checkpoint-path is not set.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional fixed 5-letter target word; random if omitted.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--max-reasoning-chars",
        type=int,
        default=2000,
        help="Maximum reasoning characters to print per turn (<= 0 means no limit).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU id to load the checkpoint on.",
    )
    args = parser.parse_args()

    checkpoint_dir = _resolve_checkpoint(args.checkpoint_path, args.search_root)
    target = args.target.lower() if args.target else pick_word()
    if len(target) != 5 or not target.isalpha() or not target.islower():
        raise ValueError(f"Target must be a lowercase 5-letter word, got: {target!r}")

    _print_section("Checkpoint")
    print(f"Using checkpoint: {checkpoint_dir}")

    _print_section("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": f"cuda:{args.gpu}"} if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Make your first guess."},
    ]

    _print_section("Game start")
    print(f"Target word: {target.upper()}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    do_sample = args.temperature > 0.0

    solved = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                pad_token_id=pad_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

        reasoning, answer_text = _split_reasoning_and_answer(response)
        guess = _extract_guess(answer_text if answer_text else response)

        _print_section(f"Attempt {attempt}")
        if reasoning:
            if args.max_reasoning_chars > 0 and len(reasoning) > args.max_reasoning_chars:
                shown = reasoning[: args.max_reasoning_chars]
                print("Reasoning:")
                print(shown)
                print(f"... (truncated {len(reasoning) - len(shown)} chars)")
            else:
                print("Reasoning:")
                print(reasoning)
        else:
            print("Reasoning: (none found)")

        print(f"Raw answer: {answer_text or response}")

        if guess is None:
            print("Could not parse a 5-letter guess; ending run.")
            break

        print(f"Action: GUESS: {guess.upper()}")
        scores = score_guess(target, guess)
        feedback = _format_feedback(guess, scores)
        print(f"Feedback: {feedback}")

        if all(s == "correct" for s in scores):
            print(f"Solved in {attempt} attempt(s).")
            solved = True
            break

        if attempt == MAX_ATTEMPTS:
            break

        messages.append({"role": "assistant", "content": response})
        messages.append(
            {
                "role": "user",
                "content": f"Feedback: {feedback}\nMake your next guess.",
            }
        )

    _print_section("Result")
    if solved:
        print("Outcome: success")
    else:
        print("Outcome: failed to solve within 6 attempts")
    print(f"Target word was: {target.upper()}")


if __name__ == "__main__":
    main()
