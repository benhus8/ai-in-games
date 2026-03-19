"""Wrapper konwertujący stare środowisko gym (0.21-0.26) do nowego API gymnasium.

slimevolleygym bazuje na starym OpenAI Gym, natomiast stable-baselines3 >= 2.0
wymaga środowisk kompatybilnych z Gymnasium. Ten moduł dostarcza klasę
GymToGymnasiumWrapper, która opakowuje stare środowisko.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces


class GymToGymnasiumWrapper(gymnasium.Env):
    """Opakowuje stare gym.Env (API 0.21+) jako gymnasium.Env."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, old_env):
        super().__init__()
        self._env = old_env

        # Przepisz przestrzenie z gym -> gymnasium
        self.observation_space = self._convert_space(old_env.observation_space)
        self.action_space = self._convert_space(old_env.action_space)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        truncated = False
        return np.array(obs, dtype=np.float32), float(reward), bool(done), truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Pomocnicze
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_space(space):
        import gym.spaces as old_spaces
        if isinstance(space, old_spaces.Box):
            return spaces.Box(
                low=space.low.astype(np.float32),
                high=space.high.astype(np.float32),
                dtype=np.float32,
            )
        if isinstance(space, old_spaces.Discrete):
            return spaces.Discrete(space.n)
        if isinstance(space, old_spaces.MultiBinary):
            return spaces.MultiBinary(space.n)
        raise NotImplementedError(f"Nieobsługiwana przestrzeń: {type(space)}")
