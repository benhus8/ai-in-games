"""
Ewaluacja checkpointów PPO w czasie – jak agent RL ewoluował?
Testuje wybrane checkpointy przeciwko:
  1. Heurystyce (nasz bot regułowy)
  2. Botowi Twórców (domyślna polityka środowiska)

Wynik: wykres win-rate vs kroki uczenia.
Uruchomienie: uv run python eval_checkpoints.py
"""
import os
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from slimevolleygym import SlimeVolleyEnv
import slimevolleygym

from agents import HeuristicPolicy, RLAgent
from eval import play_match, EvalStats


# -------------------------------------------------------
# Które checkpointy chcemy przetestować?
# Bioremy co milion kroków + kilka wczesnych
# -------------------------------------------------------
CHECKPOINT_STEPS = [
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
    7_000_000,
    10_000_000,
    15_000_000,
    20_000_000,
    25_000_000,
    30_000_000,
]
GAMES_PER_CHECKPOINT = 20  # gier na checkpoint (szybkie przybliżenie)


MAX_STEPS_PER_RALLY = 1500  # górnik kroków na rajd – po przekroczeniu rajd = remis (0:0)
RALLIES_PER_GAME = 20       # liczba rajdów zamiast "graj do N punktów"


def winrate(agent, opponent_factory, games: int) -> float:
    """Zwraca stosunek punktów zdobytych do wszystkich rozstrzygnietych rajdów.

    Używa stałej liczby rajdów z górnikiem kroków na rajd, dzięki czemu
    nieskończone wymiany (RL vs Bot Twórców) są liczone jako remis (0 pkt)
    i nie blokują skryptu.
    """
    env = SlimeVolleyEnv()
    env.policy = opponent_factory()
    our_pts = 0
    opp_pts = 0

    for _ in range(games * RALLIES_PER_GAME):
        obs = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_RALLY:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            steps += 1
        if reward > 0:
            our_pts += 1
        elif reward < 0:
            opp_pts += 1
        # reward == 0 lub timeout -> remis rajdu, 0 pkt dla nikogo

    env.close()
    total = our_pts + opp_pts
    return our_pts / total if total > 0 else 0.5


def load_agent(steps: int) -> RLAgent | None:
    path = f"agents/checkpoints/ppo_slimevolley_{steps}_steps.zip"
    if not os.path.exists(path):
        print(f"  Brak checkpointu: {path}")
        return None
    try:
        model = PPO.load(path)
        agent = RLAgent(model=model)
        return agent
    except Exception as e:
        print(f"  Błąd ładowania {path}: {e}")
        return None


def main():
    os.makedirs("results", exist_ok=True)

    steps_done = []
    wr_vs_heuristic = []
    wr_vs_creators = []

    for steps in CHECKPOINT_STEPS:
        print(f"\n=== Checkpoint: {steps:,} kroków ===")
        agent = load_agent(steps)
        if agent is None:
            continue

        wr_h = winrate(agent, HeuristicPolicy, GAMES_PER_CHECKPOINT)
        wr_c = winrate(agent, slimevolleygym.BaselinePolicy, GAMES_PER_CHECKPOINT)
        print(f"  vs Heurystyka : {wr_h:.2f}  ({int(wr_h*GAMES_PER_CHECKPOINT)}/{GAMES_PER_CHECKPOINT})")
        print(f"  vs Twórcy     : {wr_c:.2f}  ({int(wr_c*GAMES_PER_CHECKPOINT)}/{GAMES_PER_CHECKPOINT})")

        steps_done.append(steps / 1_000_000)  # oś X w milionach
        wr_vs_heuristic.append(wr_h)
        wr_vs_creators.append(wr_c)

    # -------------------------------------------------------
    # Wykres
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_done, wr_vs_heuristic, "o-", color="#e74c3c", linewidth=2,
            label="vs Heurystyka (nasz bot regułowy)")
    ax.plot(steps_done, wr_vs_creators,  "s-", color="#2ecc71", linewidth=2,
            label="vs Bot Twórców (BaselinePolicy)")

    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="50% (losowość)")
    ax.set_xlabel("Liczba kroków uczenia [mln]", fontsize=13)
    ax.set_ylabel("Win-rate", fontsize=13)
    ax.set_title("Ewolucja bota RL w czasie – win-rate na kolejnych checkpointach", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = "results/rl_learning_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nWykres zapisany → {out}")


if __name__ == "__main__":
    main()
