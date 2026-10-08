"""Configuration and errors shared by the language-model gateway."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMError(RuntimeError):
    """Raised when no backend (Groq or local) could produce a valid result."""


@dataclass(frozen=True)
class GatewayConfig:
    # task name -> Groq model id; unknown tasks use "default"
    groq_models: Mapping[str, str] = field(
        default_factory=lambda: {
            "default": "openai/gpt-oss-20b",
            "planner": "openai/gpt-oss-20b",
            "equivalence": "openai/gpt-oss-20b",
            "describe": "openai/gpt-oss-120b",
            "hard_plan": "openai/gpt-oss-120b",
        }
    )
    # Tasks whose prompt already spells out the output shape. They use plain JSON mode: sending the
    # full JSON schema as well cost about a third more prompt tokens against an 8K tokens-per-minute limit.
    json_object_tasks: frozenset[str] = frozenset({"planner", "equivalence", "expand"})
    groq_vision_model: str = "qwen/qwen3.8-27b"
    groq_whisper_model: str = "whisper-large-v3-turbo"
    local_text_model: str = "qwen3.5:4b"
    local_vision_model: str = "qwen3-vl:2b"
    ollama_host: str = "http://127.0.0.1:11434"
    reasoning_effort: str = "low"
    timeout_s: float = 30.0
    log_path: Path | None = Path("logs/llm.jsonl")
    replay_path: Path | None = None  # set together with replay_mode to record or replay structured calls
    replay_mode: str = "off"  # off | record | replay | auto (see evora.llm.replay)

    def groq_model_for(self, task: str) -> str:
        return self.groq_models.get(task, self.groq_models["default"])

    @classmethod
    def from_env(cls, env: Mapping[str, str], **overrides: object) -> GatewayConfig:
        kwargs: dict[str, object] = {}
        if env.get("OLLAMA_HOST"):
            kwargs["ollama_host"] = env["OLLAMA_HOST"]
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]
