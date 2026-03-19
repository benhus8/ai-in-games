"""Agent PPO (Proximal Policy Optimization) dla SlimeVolley przy użyciu stable-baselines3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Domyślna ścieżka do wag modelu, obok tego pliku
DEFAULT_WEIGHTS = Path(__file__).parent / "ppo_slimevolley"

# Wszystkie 8 kombinacji binarnych: [forward, backward, jump]
ACTIONS: list[list[int]] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
]
N_ACTIONS = len(ACTIONS)
OBS_DIM = 12


class RLAgent:
    """Agent PPO (Proximal Policy Optimization) dla SlimeVolley przy użyciu stable-baselines3.

    Uruchomienie w grze (main.py --agent rl):
        agent  = RLAgent()                 # ładuje wagi automatycznie
        action = agent.predict(obs)        # zwraca [forward, backward, jump]

    Trenowanie:
        uv run python train_rl.py          # domyślnie 500 000 kroków
    """

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_WEIGHTS,
        model: PPO | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)

        if model is not None:
            self.model = model
        elif self.weights_path.with_suffix(".zip").exists():
            try:
                self.model = PPO.load(str(self.weights_path))
            except Exception as e:
                print(
                    f"[RLAgent] Błąd ładowania wag: {self.weights_path} ({e})\n"
                    "  Uruchom najpierw: uv run python train_rl.py"
                )
                self.model = None
        else:
            print(
                f"[RLAgent] Brak pliku wag: {self.weights_path}.zip\n"
                "  Uruchom najpierw: uv run python train_rl.py"
            )
            self.model = None

    def predict(self, obs) -> list[int]:
        """Zwraca akcję [forward, backward, jump] — ten sam interfejs co HeuristicPolicy."""
        if self.model is None:
            return [0, 0, 0]

        # PPO z MultiBinary(3) zwraca bezpośrednio [forward, backward, jump]
        action, _ = self.model.predict(np.array(obs, dtype=np.float32), deterministic=True)
        return [int(x) for x in np.asarray(action).flat[:3]]


    def reset(self) -> None:
        pass
