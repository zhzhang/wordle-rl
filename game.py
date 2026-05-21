import random
from pathlib import Path

_WORDS_FILE = Path(__file__).parent / "words.txt"


def pick_word() -> str:
    return pick_words(1)[0]


def pick_words(n: int) -> list[str]:
    if n <= 0:
        return []
    words = [line.strip() for line in _WORDS_FILE.read_text().splitlines() if line.strip()]
    return random.choices(words, k=n)


def score_guess(word: str, guess: str) -> list[str]:
    """Score a 5-letter guess against the target word.

    Returns a list of 5 strings, one per position:
        "correct" - right letter, right position
        "present" - right letter, wrong position
        "absent"  - letter not in the word
    """
    if len(guess) != 5 or not guess.isalpha() or not guess.islower():
        raise ValueError(f"Guess must be a lowercase 5-letter word, got: {guess!r}")

    result = ["absent"] * 5
    remaining: list[str | None] = list(word)

    for i, (g, t) in enumerate(zip(guess, word)):
        if g == t:
            result[i] = "correct"
            remaining[i] = None

    for i, g in enumerate(guess):
        if result[i] == "correct":
            continue
        if g in remaining:
            result[i] = "present"
            remaining[remaining.index(g)] = None

    return result
