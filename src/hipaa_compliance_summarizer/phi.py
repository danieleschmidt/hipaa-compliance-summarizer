from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re
from typing import Dict, List, Tuple

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


@lru_cache(maxsize=None)
def _compile_pattern(expr: str) -> re.Pattern:
    """Compile regex expressions once and cache them."""
    return re.compile(expr)


@lru_cache(maxsize=1000)
def _detect_phi_cached(text: str, patterns_hash: str, patterns_tuple: Tuple[Tuple[str, str], ...]) -> Tuple[Entity, ...]:
    """Cache PHI detection results for identical text and pattern combinations.
    
    Args:
        text: The text to analyze
        patterns_hash: Hash of the pattern configuration for cache invalidation
        patterns_tuple: Tuple of (name, pattern) pairs for actual detection
        
    Returns:
        Tuple of detected entities (using tuple for hashability)
    """
    # Recreate patterns dict from tuple
    patterns = {name: _compile_pattern(expr) for name, expr in patterns_tuple}
    
    entities: List[Entity] = []
    for etype, pattern in patterns.items():
        for match in pattern.finditer(text):
            # If pattern has capturing groups, use the first group; otherwise use the full match
            if pattern.groups > 0:
                # For patterns with capturing groups, extract just the captured content
                captured_value = match.group(1)
                # Find the position of the captured group within the full match
                full_match = match.group()
                captured_start_offset = full_match.find(captured_value)
                captured_start = match.start() + captured_start_offset
                captured_end = captured_start + len(captured_value)
                entities.append(
                    Entity(etype, captured_value, captured_start, captured_end)
                )
            else:
                # For patterns without capturing groups, use the full match
                entities.append(
                    Entity(etype, match.group(), match.start(), match.end())
                )
    
    entities.sort(key=lambda e: e.start)
    return tuple(entities)


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
        self.patterns = {name: _compile_pattern(expr) for name, expr in raw_patterns.items()}
        
        # Create hashable representation for caching
        self._patterns_tuple = tuple(sorted(raw_patterns.items()))
        self._patterns_hash = hashlib.md5(str(self._patterns_tuple).encode()).hexdigest()

    def detect(self, text: str) -> List[Entity]:
        """Detect PHI entities in ``text``."""
        # Use cached detection for performance
        cached_entities = _detect_phi_cached(text, self._patterns_hash, self._patterns_tuple)
        # Convert back to list for compatibility
        return list(cached_entities)

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

    @staticmethod
    def clear_cache():
        """Clear all PHI detection caches. Useful for testing or memory management."""
        _compile_pattern.cache_clear()
        _detect_phi_cached.cache_clear()

    @staticmethod
    def get_cache_info():
        """Get information about cache usage for monitoring and debugging."""
        return {
            "pattern_compilation": _compile_pattern.cache_info(),
            "phi_detection": _detect_phi_cached.cache_info()
        }
