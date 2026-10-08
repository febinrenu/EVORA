"""Query expansion: extra wordings of a target for the image-text search.

SigLIP2 scores "a photo of a red sedan" differently from "a photo of a red car", so
searching several wordings and keeping the best score per track recovers matches the
user's word missed. Modes (config `retrieval.expansion`):

  off      the planner's own embed_text only
  lexicon  built-in synonyms, no network, no latency
  llm      the language model proposes alternatives; lexicon if it is unavailable

Every variant must keep all of the target's attributes, so expansion can widen the noun
but never drop "red" or "backpack".
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Protocol

from contracts.models import Target
from pydantic import BaseModel

from evora.llm.schemas import LLMError

log = logging.getLogger("evora.query.expand")

Mode = Literal["off", "lexicon", "llm"]
MAX_VARIANTS = 4  # including the original
MAX_TEXT_CHARS = 90

LEXICON: dict[str, list[str]] = {
    "car": ["sedan", "hatchback", "suv"],
    "suv": ["sport utility vehicle", "car"],
    "van": ["minivan", "delivery van"],
    "truck": ["lorry", "pickup truck"],
    "bus": ["coach"],
    "motorcycle": ["motorbike", "scooter"],
    "bicycle": ["bike"],
    "person": ["pedestrian"],
    "vehicle": ["car", "truck"],
}


class Expansion(BaseModel):
    texts: list[str] = []


class Gateway(Protocol):
    async def chat_json(self, task: str, messages: list[dict], schema: type[Expansion]) -> Expansion: ...


def _word_in(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word.replace('_', ' '))}\b", text.lower()) is not None


def keeps_attributes(text: str, target: Target) -> bool:
    return all(_word_in(text, a) for a in target.attributes)


def lexicon_variants(target: Target) -> list[str]:
    noun = target.noun.lower()
    out: list[str] = []
    for synonym in LEXICON.get(noun, []):
        if _word_in(target.embed_text, noun):
            out.append(re.sub(rf"\b{re.escape(noun)}\b", synonym, target.embed_text, count=1, flags=re.I))
    return out


def _clean(text: str) -> str | None:
    text = " ".join(text.split()).strip(" .\"'")
    if not text or len(text) > MAX_TEXT_CHARS:
        return None
    return text if text.lower().startswith("a photo of") else f"a photo of {text}"


def _merge(original: str, candidates: list[str], target: Target) -> list[str]:
    seen = {original.lower()}
    out = [original]
    for cand in candidates:
        cleaned = _clean(cand)
        if cleaned is None or cleaned.lower() in seen or not keeps_attributes(cleaned, target):
            continue
        seen.add(cleaned.lower())
        out.append(cleaned)
        if len(out) >= MAX_VARIANTS:
            break
    return out


def build_messages(target: Target) -> list[dict]:
    return [
        {"role": "system", "content": (
            "You rewrite a short visual description into alternative wordings for an image search. "
            'Reply only with JSON {"texts": [...]} holding up to 3 alternatives. Each starts with "a photo of", '
            "keeps every colour and attribute of the original, and changes only the wording of the object "
            "(for example car -> sedan). No new attributes, no places.")},
        {"role": "user", "content": f"Original: {target.embed_text}"},
    ]


async def expand(target: Target, mode: Mode = "off", gateway: Gateway | None = None) -> list[str]:
    """The original embed_text first, then up to MAX_VARIANTS - 1 alternatives."""
    original = target.embed_text
    if mode == "off" or not original:
        return [original] if original else []
    if mode == "llm" and gateway is not None:
        try:
            result = await gateway.chat_json("expand", build_messages(target), Expansion)
            variants = _merge(original, result.texts, target)
            if len(variants) > 1:
                return variants
        except LLMError as exc:
            log.info("llm expansion unavailable (%s); using the lexicon", exc)
    return _merge(original, lexicon_variants(target), target)
