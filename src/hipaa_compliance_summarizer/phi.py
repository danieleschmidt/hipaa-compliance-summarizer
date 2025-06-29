from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

from .config import CONFIG


@dataclass
class Entity:
    """Represents a detected PHI entity."""

    type: str
    value: str
    start: int
    end: int


@dataclass
class RedactionResult:
    """Result of redaction process."""

    text: str
    entities: List[Entity]


class PHIRedactor:
    """Simple PHI detection and redaction utility."""

    def __init__(self, mask: str = "[REDACTED]", patterns: Dict[str, str] | None = None) -> None:
        self.mask = mask
        raw_patterns = patterns or CONFIG.get("patterns", {})
        if not raw_patterns:
            raw_patterns = {
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "phone": r"\b\d{3}[.-]\d{3}[.-]\d{4}\b",
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                "date": r"\b\d{2}/\d{2}/\d{4}\b",
            }
        self.patterns = {name: re.compile(expr) for name, expr in raw_patterns.items()}

    def detect(self, text: str) -> List[Entity]:
        """Detect PHI entities in ``text``."""
        entities: List[Entity] = []
        for etype, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append(
                    Entity(etype, match.group(), match.start(), match.end())
                )
        entities.sort(key=lambda e: e.start)
        return entities

    def redact(self, text: str) -> RedactionResult:
        """Redact detected PHI from ``text`` and return result."""
        entities = self.detect(text)
        redacted_text = text
        offset = 0
        for ent in entities:
            start = ent.start + offset
            end = ent.end + offset
            redacted_text = redacted_text[:start] + self.mask + redacted_text[end:]
            offset += len(self.mask) - (end - start)
        return RedactionResult(redacted_text, entities)

    def redact_file(self, path: str, *, chunk_size: int = 4096) -> RedactionResult:
        """Redact PHI from a file by streaming its contents."""
        text_parts: list[str] = []
        entities: list[Entity] = []
        pos = 0
        with open(path, "r") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), ""):
                chunk_result = self.redact(chunk)
                text_parts.append(chunk_result.text)
                for ent in chunk_result.entities:
                    entities.append(
                        Entity(ent.type, ent.value, ent.start + pos, ent.end + pos)
                    )
                pos += len(chunk)
        combined = "".join(text_parts)
        return RedactionResult(combined, entities)
