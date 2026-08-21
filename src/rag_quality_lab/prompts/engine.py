"""Versioned prompt variants with stable content hashes."""

import hashlib
from typing import Literal

PromptVariant = Literal["direct", "evidence_first"]
PROMPT_VERSION = "1.0.0"
PROMPTS: dict[PromptVariant, str] = {
    "direct": (
        "Answer concisely using only the supplied context. Return JSON with answer, "
        "citations, and abstained. Set abstained to true when context is insufficient."
    ),
    "evidence_first": (
        "Identify supporting document citations before composing a concise answer. "
        "Use only the supplied context and return JSON with answer, citations, and "
        "abstained. Set abstained to true when context is insufficient."
    ),
}


class PromptEngine:
    """Expose canonical instructions and hashes for supported variants."""

    def instructions(self, variant: PromptVariant) -> str:
        return PROMPTS[variant]

    def prompt_hash(self, variant: PromptVariant) -> str:
        canonical = f"{PROMPT_VERSION}\n{PROMPTS[variant]}"
        return hashlib.sha256(canonical.encode()).hexdigest()

    def hashes(self) -> dict[str, str]:
        return {variant: self.prompt_hash(variant) for variant in PROMPTS}
