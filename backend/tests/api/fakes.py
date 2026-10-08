"""Test doubles shared by the query and end-to-end tests."""
from __future__ import annotations

import json

from contracts.models import QueryPlan, Referent, Target

from evora.llm.schemas import LLMError

PLACES = ("main gate", "front gate", "gate at the entrance", "loading dock", "lobby")


def plan_for(text: str) -> QueryPlan:
    place = next((p for p in PLACES if p in text.lower()), None)
    return QueryPlan(
        intent="exists",
        targets=[Target(noun="car", cls=["car"], attributes=[], embed_text="a photo of a car")],
        place=Referent(text=place, role="place") if place else None,
        action="pass_through" if place else "any",
        source="llm",
    )


class EquivalenceAnswer:
    def __init__(self, same: bool):
        self.same = same


class FakeGateway:
    """Stands in for evora.llm.gateway.Gateway: plans from the question text, never touches the network."""

    def __init__(self, same: bool = True, planner_error: bool = False, transcript: str = "red car at the main gate"):
        self.same, self.planner_error, self.transcript = same, planner_error, transcript
        self.planner_calls = 0
        self.equivalence_calls = 0

    async def chat_json_ex(self, task, messages, schema):
        self.planner_calls += 1
        if self.planner_error:
            raise LLMError("no backend available")
        question = messages[-1]["content"].split("QUESTION:", 1)[-1]
        return plan_for(question), "groq"

    async def chat_json(self, task, messages, schema):
        self.equivalence_calls += 1
        return EquivalenceAnswer(self.same)

    async def vision_yesno(self, image_jpeg, questions):
        return [None] * len(questions)

    async def transcribe(self, audio: bytes) -> str:
        if not audio.startswith(b"ok"):
            raise LLMError("transcription unavailable")
        return self.transcript


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """[(event, data)] from a text/event-stream body."""
    events, name = [], None
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = None
    return events
