"""Loading of the keyword / phrase / regex configuration file (JSON or YAML)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PatternSpec:
    name: str
    regex: str
    description: str = ""
    compiled: re.Pattern | None = None


@dataclass
class MatchConfig:
    keywords: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    patterns: list[PatternSpec] = field(default_factory=list)
    fuzzy_threshold: float = 80.0
    min_fuzzy_length: int = 4
    min_word_confidence: float = 0.0
    low_confidence_threshold: float = 60.0
    min_text_coverage: float = 0.5

    @property
    def terms(self) -> list[str]:
        """All keywords and phrases, de-duplicated, in file order."""
        seen, out = set(), []
        for term in self.keywords + self.phrases:
            key = term.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(term.strip())
        return out

    def to_dict(self) -> dict:
        return {
            "keywords": self.keywords,
            "phrases": self.phrases,
            "patterns": {p.name: {"regex": p.regex, "description": p.description} for p in self.patterns},
            "settings": {
                "fuzzy_threshold": self.fuzzy_threshold,
                "min_fuzzy_length": self.min_fuzzy_length,
                "min_word_confidence": self.min_word_confidence,
                "low_confidence_threshold": self.low_confidence_threshold,
                "min_text_coverage": self.min_text_coverage,
            },
        }


class ConfigError(ValueError):
    pass


def config_from_dict(data: dict) -> MatchConfig:
    if not isinstance(data, dict):
        raise ConfigError("Configuration must be a JSON/YAML object")
    patterns = []
    for name, spec in (data.get("patterns") or {}).items():
        if isinstance(spec, str):
            spec = {"regex": spec}
        try:
            compiled = re.compile(spec["regex"], re.IGNORECASE)
        except (KeyError, re.error) as exc:
            raise ConfigError(f"Invalid regex for pattern '{name}': {exc}") from exc
        patterns.append(PatternSpec(name, spec["regex"], spec.get("description", ""), compiled))

    settings = data.get("settings") or {}
    return MatchConfig(
        keywords=[str(k) for k in data.get("keywords", [])],
        phrases=[str(p) for p in data.get("phrases", [])],
        patterns=patterns,
        fuzzy_threshold=float(settings.get("fuzzy_threshold", 80)),
        min_fuzzy_length=int(settings.get("min_fuzzy_length", 4)),
        min_word_confidence=float(settings.get("min_word_confidence", 0)),
        low_confidence_threshold=float(settings.get("low_confidence_threshold", 60)),
        min_text_coverage=float(settings.get("min_text_coverage", 0.5)),
    )


def load_config(path: str | Path) -> MatchConfig:
    """Read a .json, .yaml or .yml configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return config_from_dict(data)
