"""Single source of truth: YAML + dotted --set overrides.

Every CLI in this repo takes `--config` and any number of `--set key.path=value`,
so an ablation row is a diff against base.yaml rather than a new script.
"""
import argparse
import copy
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access, so cfg.detector.imgsz reads like code."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _coerce(s: str):
    """YAML-parse a --set value so '0.5', 'true', '[1,2]' arrive as the right type."""
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load(config: str, sets=(), base: str | None = None) -> Cfg:
    """Load `config`; if it declares `_base`, load that first and overlay."""
    p = Path(config)
    raw = yaml.safe_load(p.read_text()) or {}
    parent = raw.pop("_base", base)
    if parent:
        pp = (p.parent / parent).resolve()
        merged = yaml.safe_load(Path(pp).read_text()) or {}
        raw = _deep_update(merged, raw)
    for s in sets:
        if "=" not in s:
            raise SystemExit(f"--set needs key=value, got {s!r}")
        key, val = s.split("=", 1)
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(val)
    return Cfg(copy.deepcopy(raw))


def add_args(ap: argparse.ArgumentParser, default="configs/base.yaml"):
    ap.add_argument("--config", default=default)
    ap.add_argument("--set", nargs="*", default=[], dest="sets",
                    help="dotted overrides, e.g. --set train.epochs=5 model.neck.p2=true")
    return ap
