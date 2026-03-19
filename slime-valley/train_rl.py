#!/usr/bin/env python
"""Trenuj agenta PPO za pomocą stable-baselines3 przeciwko wbudowanemu botowi SlimeVolley.

Użycie:
    uv run python train_rl.py
    uv run python train_rl.py --episodes 10000
    uv run python train_rl.py --weights agents/ppo_slimevolley

Po trenowaniu uruchom grę:
    uv run python main.py --agent rl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slimevolleygym import SlimeVolleyEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from agents.rl_agent import DEFAULT_WEIGHTS
from agents.gym_compat import GymToGymnasiumWrapper


def make_env():
    """Tworzy owinięte środowisko SlimeVolley (stary gym → gymnasium)."""
    env = SlimeVolleyEnv()
    return GymToGymnasiumWrapper(env)


def train(
    total_timesteps: int = 500_000,
    save_every: int = 50_000,
    weights_path: str | None = None,
    reset: bool = False,
) -> None:
    num_envs = 4
    env = DummyVecEnv([make_env for _ in range(num_envs)])

    save_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS

    # Callback do zapisu checkpointów w trakcie uczenia
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, save_every // num_envs),
        save_path=str(save_path.parent / "checkpoints"),
        name_prefix="ppo_slimevolley",
    )

    # Definicja hiperparametrów — stosujemy ZAWSZE, czy ładujemy model czy nie
    ppo_kwargs = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.001,  # zmniejszone z 0.01 → nagrody rzadkie, entropia dominowała
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )

    if save_path.with_suffix(".zip").exists() and not reset:
        print(f"Ładowanie istniejącego modelu z {save_path} (z nowymi hiperparametrami)")
        # custom_objects nadpisuje zapisane hiperparametry
        model = PPO.load(
            str(save_path),
            env=env,
            custom_objects={k: v for k, v in ppo_kwargs.items() if k != "verbose"},
        )
        model.set_parameters(model.get_parameters())  # upewnij się, że parametry są spójne
    else:
        if reset:
            print("Tryb --reset: tworzenie modelu od zera")
        else:
            print("Tworzenie nowego modelu PPO (MlpPolicy)")
        model = PPO("MlpPolicy", env, **ppo_kwargs)


    print(f"Urządzenie : {model.device}")
    print(f"Zapis do   : {save_path}")
    print(f"Kroki      : {total_timesteps}")
    print("Uruchamiam trening...\n")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save(str(save_path))
    print(f"\nGotowe! Model zapisany w: {save_path}.zip")
    print("Uruchom grę: uv run python main.py --agent rl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trenuj agenta PPO dla SlimeVolley")
    parser.add_argument(
        "--episodes",
        type=int,
        default=500_000,
        help="Liczba timestepów trenowania (domyślnie 500 000)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50_000,
        help="Zapisuj checkpoint co N kroków (domyślnie 50 000)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Ścieżka zapisu modelu (domyślnie agents/ppo_slimevolley)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Zacznij trening od zera (ignoruje istniejący model)",
    )
    args = parser.parse_args()
    train(args.episodes, args.save_every, args.weights, args.reset)


if __name__ == "__main__":
    main()
