"""
PHI Pattern Configuration Module

This module provides a modular system for managing PHI detection patterns,
including validation, loading, and extensibility for custom patterns.
"""

from __future__ import annotations

import re
import logging
import hashlib
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Union, Pattern
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PHIPatternConfig:
    """Configuration for a single PHI detection pattern."""
    
    name: str
    pattern: str
    description: str = ""
    confidence_threshold: float = 0.95
    category: str = "general"
    enabled: bool = True
    compiled_pattern: Optional[Pattern] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate and compile the pattern after initialization."""
        self.validate()
        self.compile()
    
    def validate(self) -> None:
        """Validate the pattern configuration."""
        if not self.name:
            raise ValueError("Pattern name cannot be empty")
        
        if not self.pattern:
            raise ValueError(f"Pattern string cannot be empty for pattern '{self.name}'")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0 for pattern '{self.name}'")
        
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{self.pattern}' for pattern '{self.name}': {e}")
    
    def compile(self) -> None:
        """Compile the regex pattern for performance."""
        try:
            self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Failed to compile pattern '{self.name}': {e}")
            raise


@dataclass
class PHIPatternCategory:
    """Represents a category of PHI patterns."""
    
    name: str
    description: str = ""
    patterns: Dict[str, PHIPatternConfig] = field(default_factory=dict)
    
    def add_pattern(self, pattern: PHIPatternConfig) -> None:
        """Add a pattern to this category."""
        pattern.category = self.name
        self.patterns[pattern.name] = pattern
    
    def get_enabled_patterns(self) -> Dict[str, PHIPatternConfig]:
        """Get all enabled patterns in this category."""
        return {name: pattern for name, pattern in self.patterns.items() if pattern.enabled}


class PHIPatternManager:
    """Manages PHI detection patterns with modular configuration support."""
    
    def __init__(self):
        self.categories: Dict[str, PHIPatternCategory] = {}
        self._default_patterns_loaded = False
        self._pattern_hash_cache = {}  # Cache for pattern set hashes
        self._validation_cache = {}    # Cache for validation results
        self._stats_cache = {"data": None, "timestamp": 0, "ttl": 60}  # 60 second TTL
        
    def load_default_patterns(self) -> None:
        """Load the default set of PHI patterns."""
        if self._default_patterns_loaded:
            return
            
        # Core identifier patterns
        core_category = PHIPatternCategory("core", "Core PHI identifiers")
        
        core_patterns = [
            PHIPatternConfig(
                name="ssn",
                pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                description="Social Security Number",
                confidence_threshold=0.98
            ),
            PHIPatternConfig(
                name="phone", 
                pattern=r"\b\d{3}[.-]\d{3}[.-]\d{4}\b",
                description="Phone number",
                confidence_threshold=0.95
            ),
            PHIPatternConfig(
                name="email",
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                description="Email address",
                confidence_threshold=0.98
            ),
            PHIPatternConfig(
                name="date",
                pattern=r"\b\d{2}/\d{2}/\d{4}\b",
                description="Date in MM/DD/YYYY format",
                confidence_threshold=0.90
            )
        ]
        
        for pattern in core_patterns:
            core_category.add_pattern(pattern)
        
        # Medical identifier patterns
        medical_category = PHIPatternCategory("medical", "Medical identifiers")
        
        medical_patterns = [
            PHIPatternConfig(
                name="mrn",
                pattern=r"\b(?:MRN|Medical Record|Patient ID)[:.]?\s*([A-Z]{0,3}\d{6,12})\b",
                description="Medical Record Number",
                confidence_threshold=0.95
            ),
            PHIPatternConfig(
                name="dea",
                pattern=r"\b(?:DEA|DEA#|DEA Number)[:.]?\s*([A-Z]{2}\d{7})\b", 
                description="DEA Number",
                confidence_threshold=0.98
            ),
            PHIPatternConfig(
                name="insurance_id",
                pattern=r"\b(?:Member ID|Policy|Insurance ID|Subscriber ID)[:.]?\s*([A-Z0-9]{8,15})\b",
                description="Insurance ID",
                confidence_threshold=0.92
            )
        ]
        
        for pattern in medical_patterns:
            medical_category.add_pattern(pattern)
        
        self.categories["core"] = core_category
        self.categories["medical"] = medical_category
        self._default_patterns_loaded = True
        
        logger.info(f"Loaded {len(core_patterns)} core patterns and {len(medical_patterns)} medical patterns")
    
    def load_patterns_from_config(self, config: Dict) -> None:
        """Load patterns from configuration dictionary."""
        patterns_config = config.get("patterns", {})
        
        for pattern_name, pattern_expr in patterns_config.items():
            # Determine category based on pattern name or use default
            category_name = self._determine_category(pattern_name)
            
            if category_name not in self.categories:
                self.categories[category_name] = PHIPatternCategory(category_name)
            
            pattern_config = PHIPatternConfig(
                name=pattern_name,
                pattern=pattern_expr,
                description=f"Pattern for {pattern_name}",
                category=category_name
            )
            
            self.categories[category_name].add_pattern(pattern_config)
        
        logger.info(f"Loaded {len(patterns_config)} patterns from configuration")
    
    def load_patterns_from_file(self, file_path: Union[str, Path]) -> None:
        """Load patterns from a YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Pattern file not found: {file_path}")
        
        try:
            with path.open('r') as f:
                config = yaml.safe_load(f) or {}
            self.load_patterns_from_config(config)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in pattern file {file_path}: {e}")
    
    def add_custom_pattern(self, pattern: PHIPatternConfig, category: str = "custom") -> None:
        """Add a custom PHI pattern."""
        if category not in self.categories:
            self.categories[category] = PHIPatternCategory(category, f"Custom {category} patterns")
        
        self.categories[category].add_pattern(pattern)
        logger.info(f"Added custom pattern '{pattern.name}' to category '{category}'")
    
    def get_all_patterns(self) -> Dict[str, PHIPatternConfig]:
        """Get all enabled patterns across all categories."""
        all_patterns = {}
        for category in self.categories.values():
            all_patterns.update(category.get_enabled_patterns())
        return all_patterns
    
    def get_patterns_by_category(self, category: str) -> Dict[str, PHIPatternConfig]:
        """Get all enabled patterns in a specific category."""
        if category not in self.categories:
            return {}
        return self.categories[category].get_enabled_patterns()
    
    @lru_cache(maxsize=32)
    def get_compiled_patterns_cached(self, patterns_hash: str) -> Dict[str, Pattern]:
        """Get compiled regex patterns with LRU caching based on pattern hash."""
        patterns = self.get_all_patterns()
        return {name: pattern.compiled_pattern for name, pattern in patterns.items() 
                if pattern.compiled_pattern is not None}
    
    def get_compiled_patterns(self) -> Dict[str, Pattern]:
        """Get compiled regex patterns for performance."""
        # Create hash of current pattern configuration for cache key
        patterns = self.get_all_patterns()
        pattern_repr = str(sorted([(name, config.pattern, config.enabled) 
                                  for name, config in patterns.items()]))
        patterns_hash = hashlib.md5(pattern_repr.encode(), usedforsecurity=False).hexdigest()
        
        return self.get_compiled_patterns_cached(patterns_hash)
    
    def disable_pattern(self, pattern_name: str) -> bool:
        """Disable a specific pattern by name."""
        for category in self.categories.values():
            if pattern_name in category.patterns:
                category.patterns[pattern_name].enabled = False
                # Clear caches when pattern state changes
                self._clear_pattern_caches()
                logger.info(f"Disabled pattern '{pattern_name}'")
                return True
        return False
    
    def enable_pattern(self, pattern_name: str) -> bool:
        """Enable a specific pattern by name."""
        for category in self.categories.values():
            if pattern_name in category.patterns:
                category.patterns[pattern_name].enabled = True
                # Clear caches when pattern state changes
                self._clear_pattern_caches()
                logger.info(f"Enabled pattern '{pattern_name}'")
                return True
        return False
    
    def validate_all_patterns(self) -> List[str]:
        """Validate all patterns and return any validation errors with caching."""
        # Create cache key from current pattern state
        pattern_repr = str(sorted([(name, config.pattern) 
                                  for category in self.categories.values()
                                  for name, config in category.patterns.items()]))
        validation_key = hashlib.md5(pattern_repr.encode(), usedforsecurity=False).hexdigest()
        
        # Check cache first
        if validation_key in self._validation_cache:
            return self._validation_cache[validation_key]
        
        errors = []
        for category_name, category in self.categories.items():
            for pattern_name, pattern in category.patterns.items():
                try:
                    pattern.validate()
                except ValueError as e:
                    errors.append(f"Category '{category_name}', Pattern '{pattern_name}': {e}")
        
        # Cache the result
        self._validation_cache[validation_key] = errors
        return errors
    
    def get_pattern_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded patterns with TTL caching."""
        current_time = time.time()
        
        # Check if cached data is still valid
        if (self._stats_cache["data"] is not None and 
            current_time - self._stats_cache["timestamp"] < self._stats_cache["ttl"]):
            return self._stats_cache["data"]
        
        # Calculate fresh statistics
        stats = {
            "total_categories": len(self.categories),
            "total_patterns": 0,
            "enabled_patterns": 0,
            "disabled_patterns": 0
        }
        
        for category in self.categories.values():
            stats["total_patterns"] += len(category.patterns)
            stats["enabled_patterns"] += len(category.get_enabled_patterns())
            stats["disabled_patterns"] += len(category.patterns) - len(category.get_enabled_patterns())
        
        # Cache the result with timestamp
        self._stats_cache["data"] = stats
        self._stats_cache["timestamp"] = current_time
        
        return stats
    
    def _determine_category(self, pattern_name: str) -> str:
        """Determine the appropriate category for a pattern based on its name."""
        medical_keywords = ["mrn", "dea", "insurance", "patient", "medical", "npi"]
        core_keywords = ["ssn", "phone", "email", "date", "address"]
        
        name_lower = pattern_name.lower()
        
        if any(keyword in name_lower for keyword in medical_keywords):
            return "medical"
        elif any(keyword in name_lower for keyword in core_keywords):
            return "core"
        else:
            return "custom"
    
    @lru_cache(maxsize=16)
    def get_patterns_by_category_cached(self, category: str, patterns_hash: str) -> Dict[str, PHIPatternConfig]:
        """Get enabled patterns for a category with caching based on configuration hash."""
        if category not in self.categories:
            return {}
        return self.categories[category].get_enabled_patterns()
    
    def get_patterns_by_category(self, category: str) -> Dict[str, PHIPatternConfig]:
        """Get all enabled patterns in a specific category with caching."""
        # Create hash for cache invalidation
        if category in self.categories:
            category_patterns = self.categories[category].patterns
            pattern_repr = str(sorted([(name, config.enabled, config.pattern) 
                                     for name, config in category_patterns.items()]))
            patterns_hash = hashlib.md5(pattern_repr.encode(), usedforsecurity=False).hexdigest()
            return self.get_patterns_by_category_cached(category, patterns_hash)
        return {}
    
    def _clear_pattern_caches(self) -> None:
        """Clear all pattern-related caches when patterns are modified."""
        self.get_compiled_patterns_cached.cache_clear()
        self.get_patterns_by_category_cached.cache_clear()
        self._validation_cache.clear()
        self._stats_cache["data"] = None
        logger.debug("Cleared pattern caches after configuration change")
    
    def clear_all_caches(self) -> None:
        """Clear all caches manually. Useful for testing or memory management."""
        self._clear_pattern_caches()
        logger.info("All pattern manager caches cleared")
    
    def get_cache_info(self) -> Dict[str, Dict]:
        """Get cache statistics for monitoring and debugging."""
        return {
            "compiled_patterns": self.get_compiled_patterns_cached.cache_info()._asdict(),
            "category_patterns": self.get_patterns_by_category_cached.cache_info()._asdict(),
            "validation_cache_size": len(self._validation_cache),
            "stats_cache_valid": (self._stats_cache["data"] is not None and 
                                time.time() - self._stats_cache["timestamp"] < self._stats_cache["ttl"])
        }


# Global pattern manager instance
pattern_manager = PHIPatternManager()