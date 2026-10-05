"""Experiment configuration: one YAML file per experiment, with command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .rewards import RewardSettings


def _parse_scalar(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `section.key=value` overrides. Values are parsed as YAML scalars."""
    config = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override {item!r} is not of the form section.key=value")
        dotted, raw = item.split("=", 1)
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {dotted!r}: {key!r} is not a section")
        node[leaf] = _parse_scalar(raw)
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML experiment file and apply overrides."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    config = apply_overrides(config, overrides)
    config["config_path"] = str(path)
    return config


def reward_spec(config: dict[str, Any]) -> tuple[dict[str, float], RewardSettings]:
    """Reward weights (in file order) and shaping settings from the `rewards` section."""
    section = config.get("rewards") or {}
    weights = {str(name): float(weight) for name, weight in (section.get("weights") or {}).items()}
    if not weights:
        raise ValueError("The config has no rewards.weights section")
    return weights, RewardSettings(**(section.get("settings") or {}))
