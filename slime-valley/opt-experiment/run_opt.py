"""
Izolowany eksperyment optymalizacji parametrów HeuristicPolicy.
Wyniki zapisuje WYŁĄCZNIE do opt-experiment/ – nie nadpisuje agents/opt_weights.json.

Uruchomienie:
    uv run python opt-experiment/run_opt.py

Parametry dobrane tak, by całość skończyła się w ~10-15 minut:
  - DE : popsize=6, maxiter=30, games=5 na ewaluację
  - CMA-ES: maxiter=25, games=5 na ewaluację
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Dorzucamy katalog nadrzędny do ścieżki, by importować z agents/
sys.path.insert(0, str(Path(__file__).parent.parent))

from slimevolleygym import SlimeVolleyEnv
from agents.heuristic import HeuristicPolicy
from agents.opt_agent import PARAM_BOUNDS, PARAM_NAMES, config_from_theta


OUT_DIR = Path(__file__).parent
GAMES_PER_EVAL = 8          # więcej gier = mniej szumu w fitnessu
POINTS_TO_WIN  = 3          # krócej = szybciej


# ---------------------------------------------------------------------------
# Funkcja celu (wspólna)
# ---------------------------------------------------------------------------

def evaluate(theta: list[float] | np.ndarray) -> float:
    """Zwraca ujemny win-rate (minimalizujemy).
    
    Gramy zoptymalizowanym agentem PRZECIWKO domyślnej Heurystyce (nie Botowi
    Twórców). Dzięki temu optymalizacja ma realny sygnał do nauki –
    nawet słabe parametry wygrywają część meczów.
    """
    cfg     = config_from_theta(list(theta))
    agent   = HeuristicPolicy(cfg)
    opponent = HeuristicPolicy()   # domyślne parametry jako przeciwnik
    env     = SlimeVolleyEnv()
    env.policy = opponent

    wins = 0
    for _ in range(GAMES_PER_EVAL):
        obs   = env.reset()
        our_p = opp_p = 0
        agent.reset()
        while max(our_p, opp_p) < POINTS_TO_WIN:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                if reward > 0:   our_p += 1
                elif reward < 0: opp_p += 1
                obs = env.reset()
                agent.reset()
        if our_p > opp_p:
            wins += 1

    env.close()
    return -wins / GAMES_PER_EVAL


# ---------------------------------------------------------------------------
# Differential Evolution
# ---------------------------------------------------------------------------

def run_de(maxiter: int = 30, popsize: int = 6, seed: int = 42):
    from scipy.optimize import differential_evolution
    history: list[float] = []

    def callback(xk, convergence):
        wr = -evaluate(xk)
        history.append(wr)
        print(f"  [DE]  gen {len(history):3d}  win-rate={wr:.3f}  conv={convergence:.4f}")

    print(f"\n{'='*55}")
    print(f"  Differential Evolution  (pop={popsize}, maxiter={maxiter}, games={GAMES_PER_EVAL})")
    print(f"{'='*55}")
    t0 = time.perf_counter()

    result = differential_evolution(
        evaluate,
        bounds=PARAM_BOUNDS,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.9,
        callback=callback,
        workers=1,
        disp=False,
    )
    elapsed = time.perf_counter() - t0
    best_wr = -result.fun
    print(f"\n  DE zakończone: win-rate={best_wr:.3f}  czas={elapsed:.1f}s")
    return list(result.x), history


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def run_cmaes(maxiter: int = 25, sigma0: float = 0.3, seed: int = 42):
    import cma
    history: list[float] = []

    bounds_lo = np.array([b[0] for b in PARAM_BOUNDS])
    bounds_hi = np.array([b[1] for b in PARAM_BOUNDS])

    def normalize(t):   return (np.array(t) - bounds_lo) / (bounds_hi - bounds_lo)
    def denormalize(t): return bounds_lo + np.array(t) * (bounds_hi - bounds_lo)

    from agents.opt_agent import theta_from_config
    x0_n = normalize(theta_from_config()).tolist()

    opts = cma.CMAOptions()
    opts["maxiter"] = maxiter
    opts["bounds"]  = [[0.0]*len(x0_n), [1.0]*len(x0_n)]
    opts["seed"]    = seed
    opts["verbose"] = -9

    print(f"\n{'='*55}")
    print(f"  CMA-ES  (sigma0={sigma0}, maxiter={maxiter}, games={GAMES_PER_EVAL})")
    print(f"{'='*55}")
    t0 = time.perf_counter()

    es  = cma.CMAEvolutionStrategy(x0_n, sigma0, opts)
    gen = 0
    while not es.stop():
        sols_n  = es.ask()
        fitvals = [evaluate(denormalize(s)) for s in sols_n]
        es.tell(sols_n, fitvals)
        gen += 1
        best_wr = -min(fitvals)
        history.append(best_wr)
        print(f"  [CMA-ES]  gen {gen:3d}  win-rate(best)={best_wr:.3f}  sigma={es.sigma:.4f}")

    elapsed  = time.perf_counter() - t0
    best_wr  = -es.result.fbest
    print(f"\n  CMA-ES zakończone: win-rate={best_wr:.3f}  czas={elapsed:.1f}s")
    return list(denormalize(es.result.xbest)), history


# ---------------------------------------------------------------------------
# Wykres
# ---------------------------------------------------------------------------

def plot_convergence(histories: dict[str, list[float]], out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = {"de": "#e74c3c", "cmaes": "#2ecc71"}
    labels  = {"de": "Differential Evolution (DE)", "cmaes": "CMA-ES"}

    for method, hist in histories.items():
        if hist:
            ax.plot(hist, label=labels.get(method, method),
                    color=colors.get(method, None), linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Generacja / Iteracja", fontsize=12)
    ax.set_ylabel("Win-rate najlepszego osobnika", fontsize=12)
    ax.set_title("Konwergencja optymalizacji parametrów heurystyki", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n  Wykres → {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    _, h_de    = run_de(maxiter=30, popsize=6, seed=42)
    _, h_cmaes = run_cmaes(maxiter=25, sigma0=0.3, seed=42)

    histories = {"de": h_de, "cmaes": h_cmaes}

    # Zapis surowych danych
    hist_path = OUT_DIR / "opt_history_real.json"
    hist_path.write_text(json.dumps(histories, indent=2))
    print(f"  Historia → {hist_path}")

    # Wykres
    plot_convergence(histories, OUT_DIR / "convergence_real.png")


if __name__ == "__main__":
    main()
