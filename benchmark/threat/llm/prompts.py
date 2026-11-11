"""Prompt and output schema for the zero-shot LLM reconstruction attacker.

Policy-blind by design: the prompt never says what kind of text was removed, so the
same prompt is valid for gold, leave-one-out and random-word masks. The mask token
carries neither length nor type.
"""
from __future__ import annotations

SYSTEM = (
    "You are given a product review in which some words or short phrases have been removed "
    "and replaced with numbered placeholders such as [MASK_1], [MASK_2]. Each placeholder "
    "replaces one word or one short phrase; the placeholder gives no hint about its length.\n\n"
    "Your task: for every placeholder, infer the exact text that was removed, using every clue "
    "in the surrounding review. Give your five best guesses per placeholder, ordered from most "
    "to least likely, each written as the literal removed text (not a description of it), and a "
    "confidence between 0 and 1 that your first guess is exactly right. Always give exactly five "
    "distinct guesses for every placeholder, even when unsure."
)

USER_TEMPLATE = "Review:\n\n{masked_text}\n\nPlaceholders to fill: {placeholders}."

SCHEMA = {
    "type": "object",
    "properties": {
        "guesses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mask": {"type": "integer", "description": "placeholder number k of [MASK_k]"},
                    "candidates": {"type": "array", "items": {"type": "string"},
                                   "description": "five guesses, most likely first"},
                    "confidence": {"type": "number",
                                   "description": "0-1 probability that candidates[0] is exactly right"},
                },
                "required": ["mask", "candidates", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["guesses"],
    "additionalProperties": False,
}

MEMCHECK_SYSTEM = (
    "You will be shown the beginning of a product review. Some words in it have been replaced by "
    "placeholders such as [MASK_1]. If you have seen this exact review before, reproduce the next "
    "sentence of the original review word for word. If you have not, write the single most likely "
    "next sentence. Return only that one sentence."
)

MEMCHECK_USER = "Beginning of the review:\n\n{prefix_masked}\n\nNext sentence:"


def user_prompt(masked_text: str, n_masks: int) -> str:
    return USER_TEMPLATE.format(masked_text=masked_text,
                                placeholders=", ".join(f"[MASK_{k}]" for k in range(1, n_masks + 1)))
