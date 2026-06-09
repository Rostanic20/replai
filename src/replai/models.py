from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def _id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> float:
    return time.time()


@dataclass
class Span:
    run_id: str
    name: str
    type: str = "function"  # llm_call | tool_call | function
    id: str = field(default_factory=_id)
    parent_id: Optional[str] = None
    start: float = field(default_factory=_now)
    end: Optional[float] = None
    input: Any = None
    output: Any = None
    error: Optional[str] = None
    model: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    raw: Any = None  # serialized SDK response, kept for replay

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end is None:
            return None
        return round((self.end - self.start) * 1000, 1)


@dataclass
class Run:
    name: str
    id: str = field(default_factory=_id)
    start: float = field(default_factory=_now)
    end: Optional[float] = None
    metadata: dict = field(default_factory=dict)
