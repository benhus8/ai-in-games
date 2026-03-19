#!/usr/bin/env python
"""Ewaluacja agenta SlimeVolley w trybie headless (bez GUI).

Użycie:
    uv run python eval.py --agent heuristic --games 50
    uv run python eval.py --agent rl --games 100
    uv run python eval.py --agent rl --games 50 --points-to-win 5

Definicja meczu: pierwszy agent który zdobędzie --points-to-win punktów wygrywa mecz.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from slimevolleygym import SlimeVolleyEnv

from agents import HeuristicPolicy, OptimizedPolicy, RLAgent


@dataclass
class MatchResult:
    our_points: int = 0
    opp_points: int = 0
    rallies: int = 0
    duration_s: float = 0.0

    @property
    def won(self) -> bool:
        return self.our_points > self.opp_points

    @property
    def lost(self) -> bool:
        return self.opp_points > self.our_points

    @property
    def draw(self) -> bool:
        return self.our_points == self.opp_points


@dataclass
class EvalStats:
    results: list[MatchResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def wins(self) -> int:
        return sum(r.won for r in self.results)

    @property
    def losses(self) -> int:
        return sum(r.lost for r in self.results)

    @property
    def draws(self) -> int:
        return sum(r.draw for r in self.results)

    @property
    def total_rallies(self) -> int:
        return sum(r.rallies for r in self.results)

    @property
    def avg_rallies(self) -> float:
        return self.total_rallies / self.n if self.n else 0.0

    @property
    def avg_duration_s(self) -> float:
        return sum(r.duration_s for r in self.results) / self.n if self.n else 0.0

    @property
    def total_our_points(self) -> int:
        return sum(r.our_points for r in self.results)

    @property
    def total_opp_points(self) -> int:
        return sum(r.opp_points for r in self.results)


def play_match(env, agent, points_to_win: int, max_steps_per_rally: int = 3000) -> MatchResult:
    result = MatchResult()
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    t_start = time.perf_counter()
    while max(result.our_points, result.opp_points) < points_to_win:
        steps_in_rally = 0
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            steps_in_rally += 1
            if steps_in_rally >= max_steps_per_rally:
                # Rajd trwa za długo — liczymy jako remis rajdu (brak punktu)
                done = True
                reward = 0
        result.rallies += 1
        if reward > 0:
            result.our_points += 1
        elif reward < 0:
            result.opp_points += 1
        obs = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()

    result.duration_s = time.perf_counter() - t_start
    return result


def print_summary(stats: EvalStats, agent_name: str, points_to_win: int) -> None:
    bar = "═" * 50
    print(f"\n{bar}")
    print(f"  EWALUACJA: {agent_name.upper()}   ({stats.n} meczów, do {points_to_win} pkt)")
    print(bar)
    print(f"  Wygrane  : {stats.wins:4d}  ({100*stats.wins/stats.n:.1f}%)")
    print(f"  Remisy   : {stats.draws:4d}  ({100*stats.draws/stats.n:.1f}%)")
    print(f"  Przegrane: {stats.losses:4d}  ({100*stats.losses/stats.n:.1f}%)")
    print(f"  Stosunek punktów: {stats.total_our_points} : {stats.total_opp_points}"
          f"  (nas : przeciwnik)")
    print(f"────────────────────────────────────────────────────")
    print(f"  Łącznie rajdów   : {stats.total_rallies}")
    print(f"  Rajdy / mecz     : {stats.avg_rallies:.1f}")
    print(f"  Czas / mecz      : {stats.avg_duration_s:.2f}s")
    print(f"  Łączny czas      : {sum(r.duration_s for r in stats.results):.1f}s")
    print(f"{bar}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja agenta SlimeVolley (headless)")
    parser.add_argument(
        "--agent",
        choices=["heuristic", "rl", "opt"],
        default="heuristic",
        help="Który agent ma być oceniany (domyślnie: heuristic)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Liczba meczów do rozegrania (domyślnie: 50)",
    )
    parser.add_argument(
        "--points-to-win",
        type=int,
        default=5,
        help="Liczba punktów do wygrania meczu (domyślnie: 5)",
    )
    args = parser.parse_args()

    print(f"Ładowanie agenta: {args.agent}...")
    if args.agent == "heuristic":
        agent = HeuristicPolicy()
    elif args.agent == "rl":
        agent = RLAgent()
    else:
        agent = OptimizedPolicy()

    env = SlimeVolleyEnv()
    stats = EvalStats()

    print(f"Rozgrywam {args.games} meczów (pierwszy do {args.points_to_win} pkt)...\n")

    for i in range(1, args.games + 1):
        result = play_match(env, agent, args.points_to_win)
        stats.results.append(result)

        outcome = "W" if result.won else ("R" if result.draw else "L")
        print(
            f"  Mecz {i:3d}/{args.games}  [{outcome}]"
            f"  {result.our_points}:{result.opp_points}"
            f"  rajdy={result.rallies}"
            f"  t={result.duration_s:.2f}s"
        )

    env.close()
    print_summary(stats, args.agent, args.points_to_win)


if __name__ == "__main__":
    main()
