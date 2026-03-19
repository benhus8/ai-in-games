#!/usr/bin/env python
"""Optymalizacja parametrów HeuristicPolicy dla SlimeVolley.

Trzy metody:
  - DE    : Differential Evolution (scipy)
  - CMA-ES: Covariance Matrix Adaptation Evolution Strategy (cma)
  - Bayes : Bayesian Optimization z Gaussian Process (scikit-optimize)

Użycie:
    uv run python train_opt.py --method de    --games 10 --iters 200
    uv run python train_opt.py --method cmaes --games 10 --iters 100
    uv run python train_opt.py --method bayes --games 10 --iters 50
    uv run python train_opt.py --method all   --games 8  --iters 40

Wynik: agents/opt_weights.json  +  wykresy konwergencji PNG
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from slimevolleygym import SlimeVolleyEnv

from agents.heuristic import HeuristicPolicy
from agents.opt_agent import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    PARAM_SPACE,
    config_from_theta,
    theta_from_config,
)

WEIGHTS_PATH = Path("agents/opt_weights.json")


# ---------------------------------------------------------------------------
# Funkcja celu
# ---------------------------------------------------------------------------

def evaluate(theta: list[float] | np.ndarray, n_games: int = 10) -> float:
    """Zwraca ujemny win-rate dla wektora parametrów theta.

    Minimalizujemy wartość → im wyższy win-rate, tym niższa wartość funkcji.
    """
    cfg = config_from_theta(list(theta))
    agent = HeuristicPolicy(cfg)
    env = SlimeVolleyEnv()

    wins = 0
    total = 0
    points_to_win = 5

    for _ in range(n_games):
        obs = env.reset()
        agent.reset()
        our_pts = opp_pts = 0
        while max(our_pts, opp_pts) < points_to_win:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                if reward > 0:
                    our_pts += 1
                elif reward < 0:
                    opp_pts += 1
                obs = env.reset()
                agent.reset()
        total += 1
        if our_pts > opp_pts:
            wins += 1

    env.close()
    win_rate = wins / total
    return -win_rate   # minimalizujemy


# ---------------------------------------------------------------------------
# Differential Evolution
# ---------------------------------------------------------------------------

def run_de(n_games: int, maxiter: int, popsize: int, seed: int) -> tuple[list[float], list[float]]:
    from scipy.optimize import differential_evolution

    history: list[float] = []

    def callback(xk, convergence):
        score = evaluate(xk, n_games)
        history.append(-score)
        print(
            f"  [DE]  generacja {len(history):3d}  win-rate={-score:.3f}  conv={convergence:.4f}"
        )

    print(f"\n{'='*60}")
    print(f"  Differential Evolution  (popsize={popsize}, maxiter={maxiter}, games={n_games})")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    result = differential_evolution(
        evaluate,
        bounds=PARAM_BOUNDS,
        args=(n_games,),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.9,
        callback=callback,
        workers=1,
        disp=False,
    )

    elapsed = time.perf_counter() - t0
    best_theta = list(result.x)
    best_wr = -result.fun
    print(f"\n  DE zakończone: win-rate={best_wr:.3f}  czas={elapsed:.1f}s")
    return best_theta, history


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def run_cmaes(n_games: int, maxiter: int, sigma0: float, seed: int) -> tuple[list[float], list[float]]:
    import cma

    x0 = theta_from_config()
    history: list[float] = []

    print(f"\n{'='*60}")
    print(f"  CMA-ES  (sigma0={sigma0}, maxiter={maxiter}, games={n_games})")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    # Normalizacja: pracujemy w znormalizowanej przestrzeni [0,1]^d
    bounds_lo = np.array([b[0] for b in PARAM_BOUNDS])
    bounds_hi = np.array([b[1] for b in PARAM_BOUNDS])

    def normalize(theta):
        return (np.array(theta) - bounds_lo) / (bounds_hi - bounds_lo)

    def denormalize(theta_n):
        return bounds_lo + np.array(theta_n) * (bounds_hi - bounds_lo)

    x0_n = normalize(x0).tolist()

    opts = cma.CMAOptions()
    opts["maxiter"] = maxiter
    opts["bounds"] = [[0.0] * len(x0_n), [1.0] * len(x0_n)]
    opts["seed"] = seed
    opts["verbose"] = -9   # cichy tryb — będziemy sami drukować

    es = cma.CMAEvolutionStrategy(x0_n, sigma0, opts)
    gen = 0
    while not es.stop():
        solutions_n = es.ask()
        fitvals = [evaluate(denormalize(s), n_games) for s in solutions_n]
        es.tell(solutions_n, fitvals)
        gen += 1
        best_wr = -min(fitvals)
        history.append(best_wr)
        print(f"  [CMA-ES]  gen {gen:3d}  win-rate(best)={best_wr:.3f}  sigma={es.sigma:.4f}")

    elapsed = time.perf_counter() - t0
    best_theta = list(denormalize(es.result.xbest))
    best_wr = -es.result.fbest
    print(f"\n  CMA-ES zakończone: win-rate={best_wr:.3f}  czas={elapsed:.1f}s")
    return best_theta, history


# ---------------------------------------------------------------------------
# Bayesian Optimization (Gaussian Process)
# ---------------------------------------------------------------------------

def run_bayes(n_games: int, n_calls: int, seed: int) -> tuple[list[float], list[float]]:
    from skopt import gp_minimize
    from skopt.space import Real

    space = [Real(lo, hi, name=name) for name, lo, hi in PARAM_SPACE]
    history: list[float] = []

    print(f"\n{'='*60}")
    print(f"  Bayesian Optimization / GP  (n_calls={n_calls}, games={n_games})")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    def objective(theta):
        score = evaluate(theta, n_games)
        history.append(-score)
        print(f"  [Bayes]  call {len(history):3d}  win-rate={-score:.3f}")
        return score

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=max(5, n_calls // 5),
        acq_func="EI",
        random_state=seed,
        noise=0.05,
    )

    elapsed = time.perf_counter() - t0
    best_theta = list(result.x)
    best_wr = -result.fun
    print(f"\n  Bayes zakończone: win-rate={best_wr:.3f}  czas={elapsed:.1f}s")
    return best_theta, history


# ---------------------------------------------------------------------------
# Zapis wag + wykresy
# ---------------------------------------------------------------------------

def save_weights(theta: list[float], method: str) -> None:
    data = {name: float(val) for name, val in zip(PARAM_NAMES, theta)}
    data["_method"] = method
    WEIGHTS_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n  Wagi zapisane → {WEIGHTS_PATH}")
    print("  Parametry:")
    for name, val in zip(PARAM_NAMES, theta):
        print(f"    {name:30s} = {val:.4f}")


def plot_convergence(histories: dict[str, list[float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib niedostępny — pomijam wykresy")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"de": "#e74c3c", "cmaes": "#2ecc71", "bayes": "#3498db"}
    for method, hist in histories.items():
        if hist:
            ax.plot(hist, label=method.upper(), color=colors.get(method, None), linewidth=2)

    ax.set_xlabel("Iteracja / wywołanie funkcji celu")
    ax.set_ylabel("Win-rate (wyższy = lepszy)")
    ax.set_title("Konwergencja optymalizacji — SlimeVolley HeuristicPolicy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    print(f"  Wykres konwergencji → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Funkcja porównująca agentów
# ---------------------------------------------------------------------------

def compare_agents(theta_opt: list[float], n_games: int = 30) -> None:
    """Porównuje zoptymalizowanego agenta z domyślną heurystyką."""
    print(f"\n{'='*60}")
    print("  PORÓWNANIE  (domyślny vs. zoptymalizowany, {n_games} meczów)")
    print(f"{'='*60}")

    default_wr = -evaluate(theta_from_config(), n_games)
    opt_wr = -evaluate(theta_opt, n_games)

    print(f"  Domyślna heurystyka  : win-rate = {default_wr:.3f}")
    print(f"  Zoptymalizowany agent: win-rate = {opt_wr:.3f}")
    delta = opt_wr - default_wr
    arrow = "▲" if delta >= 0 else "▼"
    print(f"  Zmiana               : {arrow} {abs(delta):.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optymalizacja parametrów HeuristicPolicy")
    parser.add_argument(
        "--method",
        choices=["de", "cmaes", "bayes", "all"],
        default="de",
        help="Metoda optymalizacji (domyślnie: de)",
    )
    parser.add_argument("--games",   type=int, default=10, help="Meczów na ewaluację (domyślnie: 10)")
    parser.add_argument("--iters",   type=int, default=100, help="Maks. iteracji / wywołań (domyślnie: 100)")
    parser.add_argument("--popsize", type=int, default=12, help="DE: rozmiar populacji (domyślnie: 12)")
    parser.add_argument("--sigma0",  type=float, default=0.3, help="CMA-ES: sigma0 (domyślnie: 0.3)")
    parser.add_argument("--seed",    type=int, default=42, help="Ziarno losowości")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Porównaj wynik z domyślną heurystyką po optymalizacji")
    parser.add_argument("--no-compare", dest="compare", action="store_false")
    args = parser.parse_args()

    methods = ["de", "cmaes", "bayes"] if args.method == "all" else [args.method]
    histories: dict[str, list[float]] = {}
    best_theta = theta_from_config()
    best_method = "default"

    for method in methods:
        if method == "de":
            theta, hist = run_de(args.games, args.iters, args.popsize, args.seed)
        elif method == "cmaes":
            theta, hist = run_cmaes(args.games, args.iters, args.sigma0, args.seed)
        else:
            theta, hist = run_bayes(args.games, args.iters, args.seed)

        histories[method] = hist
        # wybierz najlepszy wynik
        if hist and (-evaluate(theta, args.games)) > (-evaluate(best_theta, args.games)):
            best_theta = theta
            best_method = method

    # Zapis najlepszych wag
    save_weights(best_theta if len(methods) == 1 else best_theta,
                 methods[0] if len(methods) == 1 else best_method)

    # Zapis historii konwergencji
    hist_path = Path("agents/opt_history.json")
    hist_path.write_text(json.dumps(histories, indent=2))

    # Wykres konwergencji
    plot_convergence(histories, Path("agents/convergence.png"))

    # Porównanie
    if args.compare:
        n_cmp = min(30, args.games * 3)
        compare_agents(best_theta, n_cmp)


if __name__ == "__main__":
    main()
