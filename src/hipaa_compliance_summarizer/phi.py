from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


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

    def __init__(self, mask: str = "[REDACTED]") -> None:
        self.mask = mask
        self.patterns = {
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "phone": re.compile(r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "date": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
        }

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
