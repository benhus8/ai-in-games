"""OptimizedPolicy — HeuristicPolicy z parametrami wczytanymi z pliku JSON.

Plik opt_weights.json jest zapisywany przez train_opt.py po zakończeniu optymalizacji.
"""

from __future__ import annotations

import json
from pathlib import Path

from .heuristic import HeuristicConfig, HeuristicPolicy

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "opt_weights.json"

# Nazwy i zakresy 12 parametrów podlegających optymalizacji.
# Każda krotka: (nazwa_pola, dolna_granica, górna_granica)
PARAM_SPACE: list[tuple[str, float, float]] = [
    ("defend_x",               8.0,  20.0),
    ("intercept_x",            6.0,  16.0),
    ("retreat_x",             14.0,  24.0),
    ("deep_retreat_x",        13.0,  23.0),
    ("edge_recenter_x",       12.0,  20.0),
    ("deep_ball_x",           13.0,  22.0),
    ("falling_ball_x",         8.0,  18.0),
    ("falling_ball_vy",       -1.0,   4.0),
    ("general_under_ball_bias", 0.0,  3.0),
    ("under_ball_bias",         0.0,  2.0),
    ("high_ball_y",             8.0,  18.0),
    ("edge_ball_x",            14.0,  22.0),
]

PARAM_NAMES = [p[0] for p in PARAM_SPACE]
PARAM_BOUNDS = [(p[1], p[2]) for p in PARAM_SPACE]


def config_from_theta(theta: list[float] | None) -> HeuristicConfig:
    """Tworzy HeuristicConfig z wektora parametrów."""
    if theta is None:
        return HeuristicConfig()
    cfg = HeuristicConfig()
    for name, value in zip(PARAM_NAMES, theta):
        setattr(cfg, name, float(value))
    return cfg


def theta_from_config(cfg: HeuristicConfig | None = None) -> list[float]:
    """Zwraca domyślny wektor parametrów (ze standardowego HeuristicConfig)."""
    cfg = cfg or HeuristicConfig()
    return [getattr(cfg, name) for name in PARAM_NAMES]


class OptimizedPolicy:
    """Agent heurystyczny z parametrami załadowanymi z opt_weights.json."""

    def __init__(self, weights_path: Path | str | None = None):
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            theta = [data[name] for name in PARAM_NAMES]
            print(f"[OptimizedPolicy] Wczytano wagi z {path}")
        else:
            print(
                f"[OptimizedPolicy] Brak pliku {path} — używam domyślnych parametrów."
            )
            theta = None
        self._policy = HeuristicPolicy(config_from_theta(theta))

    def predict(self, obs):
        return self._policy.predict(obs)

    def reset(self):
        self._policy.reset()
