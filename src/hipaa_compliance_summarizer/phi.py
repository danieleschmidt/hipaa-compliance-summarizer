from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re
import time
from typing import Dict, List, Tuple, Optional
import logging

from .config import CONFIG
from .constants import PERFORMANCE_LIMITS
from .phi_patterns import pattern_manager, PHIPatternConfig

logger = logging.getLogger(__name__)


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
    """Advanced PHI detection and redaction utility with modular pattern support."""

    def __init__(self, mask: str = "[REDACTED]", patterns: Dict[str, str] | None = None, 
                 pattern_config: Optional[PHIPatternConfig] = None,
                 performance_monitor: Optional[object] = None) -> None:
        """
        Initialize PHI redactor with flexible pattern configuration.
        
        Args:
            mask: String to replace detected PHI with
            patterns: Legacy dict of pattern name -> regex string (for backward compatibility)
            pattern_config: Optional custom pattern configuration
            performance_monitor: Optional performance monitor for metrics collection
        """
        self.mask = mask
        self.performance_monitor = performance_monitor
        
        # Initialize pattern manager if not already done
        if not pattern_manager._default_patterns_loaded:
            pattern_manager.load_default_patterns()
        
        # Load patterns from config if available
        config_patterns = CONFIG.get("patterns", {})
        if config_patterns:
            pattern_manager.load_patterns_from_config(CONFIG)
        
        # Handle legacy patterns parameter for backward compatibility
        if patterns:
            logger.info("Using legacy patterns parameter - consider migrating to modular pattern system")
            raw_patterns = patterns
        else:
            # Use patterns from the modular system
            pattern_configs = pattern_manager.get_all_patterns()
            raw_patterns = {name: config.pattern for name, config in pattern_configs.items()}
        
        # Compile patterns for performance
        self.patterns = {name: _compile_pattern(expr) for name, expr in raw_patterns.items()}
        
        # Create hashable representation for caching
        self._patterns_tuple = tuple(sorted(raw_patterns.items()))
        self._patterns_hash = hashlib.md5(str(self._patterns_tuple).encode(), usedforsecurity=False).hexdigest()
        
        logger.info(f"Initialized PHI redactor with {len(self.patterns)} patterns")

    def detect(self, text: str) -> List[Entity]:
        """Detect PHI entities in ``text``."""
        start_time = time.time() if self.performance_monitor else None
        
        # Check cache first for monitoring purposes
        cache_key = (text, self._patterns_hash, self._patterns_tuple)
        cache_hit = _detect_phi_cached.cache_info().hits
        
        # Use cached detection for performance
        cached_entities = _detect_phi_cached(text, self._patterns_hash, self._patterns_tuple)
        
        # Record performance metrics if monitor is available
        if self.performance_monitor and start_time:
            detection_time = time.time() - start_time
            was_cache_hit = _detect_phi_cached.cache_info().hits > cache_hit
            
            # Record metrics for each pattern that had matches
            entity_types = {entity.type for entity in cached_entities}
            for pattern_name in entity_types:
                # Get average confidence for this pattern from pattern manager
                pattern_configs = pattern_manager.get_all_patterns()
                # Get confidence from pattern config or use default
                pattern_config = pattern_configs.get(pattern_name)
                confidence = pattern_config.confidence_threshold if pattern_config else 0.95
                
                self.performance_monitor.record_pattern_performance(
                    pattern_name,
                    detection_time / len(entity_types),  # Distribute time across matching patterns
                    was_cache_hit,
                    confidence
                )
        
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

    def redact_file(self, path: str, *, chunk_size: int = PERFORMANCE_LIMITS.DEFAULT_READ_CHUNK_SIZE) -> RedactionResult:
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
        pattern_manager.clear_all_caches()

    @staticmethod
    def get_cache_info():
        """Get information about cache usage for monitoring and debugging."""
        phi_cache_info = {
            "pattern_compilation": _compile_pattern.cache_info(),
            "phi_detection": _detect_phi_cached.cache_info()
        }
        
        # Include pattern manager cache info
        pattern_manager_cache = pattern_manager.get_cache_info()
        phi_cache_info.update({
            "pattern_manager": pattern_manager_cache
        })
        
        return phi_cache_info
    
    def add_custom_pattern(self, name: str, pattern: str, description: str = "", 
                          confidence_threshold: float = 0.95, category: str = "custom") -> None:
        """Add a custom PHI pattern to the redactor.
        
        Args:
            name: Unique name for the pattern
            pattern: Regular expression pattern
            description: Human-readable description
            confidence_threshold: Confidence threshold (0.0 to 1.0)
            category: Pattern category for organization
        """
        phi_pattern = PHIPatternConfig(
            name=name,
            pattern=pattern,
            description=description,
            confidence_threshold=confidence_threshold,
            category=category
        )
        
        pattern_manager.add_custom_pattern(phi_pattern, category)
        
        # Update this redactor's patterns
        self._refresh_patterns()
        logger.info(f"Added custom pattern '{name}' and refreshed redactor")
    
    def disable_pattern(self, pattern_name: str) -> bool:
        """Disable a specific pattern by name.
        
        Args:
            pattern_name: Name of the pattern to disable
            
        Returns:
            True if pattern was found and disabled, False otherwise
        """
        result = pattern_manager.disable_pattern(pattern_name)
        if result:
            self._refresh_patterns()
        return result
    
    def enable_pattern(self, pattern_name: str) -> bool:
        """Enable a specific pattern by name.
        
        Args:
            pattern_name: Name of the pattern to enable
            
        Returns:
            True if pattern was found and enabled, False otherwise
        """
        result = pattern_manager.enable_pattern(pattern_name)
        if result:
            self._refresh_patterns()
        return result
    
    def list_patterns(self) -> Dict[str, Dict[str, str]]:
        """List all available patterns with their details.
        
        Returns:
            Dictionary mapping pattern names to their configuration details
        """
        patterns = pattern_manager.get_all_patterns()
        return {
            name: {
                "pattern": config.pattern,
                "description": config.description,
                "category": config.category,
                "confidence_threshold": config.confidence_threshold,
                "enabled": config.enabled
            }
            for name, config in patterns.items()
        }
    
    def get_pattern_statistics(self) -> Dict[str, int]:
        """Get statistics about the current pattern configuration."""
        return pattern_manager.get_pattern_statistics()
    
    def _refresh_patterns(self) -> None:
        """Refresh the internal pattern cache after configuration changes."""
        # Use cached compiled patterns from pattern manager
        compiled_patterns = pattern_manager.get_compiled_patterns()
        pattern_configs = pattern_manager.get_all_patterns()
        raw_patterns = {name: config.pattern for name, config in pattern_configs.items()}
        
        # Update compiled patterns using cached versions when available
        self.patterns = {}
        for name, expr in raw_patterns.items():
            if name in compiled_patterns and compiled_patterns[name] is not None:
                self.patterns[name] = compiled_patterns[name]
            else:
                self.patterns[name] = _compile_pattern(expr)
        
        # Update cache keys
        self._patterns_tuple = tuple(sorted(raw_patterns.items()))
        self._patterns_hash = hashlib.md5(str(self._patterns_tuple).encode(), usedforsecurity=False).hexdigest()
        
        # Clear relevant caches since patterns changed
        _detect_phi_cached.cache_clear()
        
        logger.debug(f"Refreshed patterns - now using {len(self.patterns)} patterns")
