import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from eval import play_match, EvalStats
from slimevolleygym import SlimeVolleyEnv
from agents import HeuristicPolicy, OptimizedPolicy, RLAgent

def run_eval(agent_name, agent, env, games=100, opponent_name="default", opponent=None):
    print(f"Running {games} games: {agent_name} vs {opponent_name}")
    if opponent:
        env.policy = opponent
    else:
        import slimevolleygym
        env.policy = slimevolleygym.BaselinePolicy()
        
    results = []
    for i in range(games):
        res = play_match(env, agent, points_to_win=5)
        results.append({
            "agent": agent_name,
            "opponent": opponent_name,
            "our_points": res.our_points,
            "opp_points": res.opp_points,
            "rallies": res.rallies,
            "duration_s": res.duration_s,
            "won": int(res.won)
        })
    return pd.DataFrame(results)

def main():
    os.makedirs("results", exist_ok=True)
    env = SlimeVolleyEnv()
    
    heur_agent = HeuristicPolicy()
    opt_agent = OptimizedPolicy()
    rl_agent = RLAgent()
    
    # Baseline
    df_heur = run_eval("Heuristic", heur_agent, env, 50)
    df_opt = run_eval("Optimized", opt_agent, env, 50)
    df_rl = run_eval("RL", rl_agent, env, 50)
    
    df_all = pd.concat([df_heur, df_opt, df_rl])
    df_all.to_csv("results/baseline.csv", index=False)
    
    # Boxplots
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_all, x="agent", y="our_points")
    plt.title("Punkty zdobyte przez agentów przeciwko domyślnemu przeciwnikowi")
    plt.savefig("results/boxplot_points.png")
    
    # Tournament
    print("Running tournament")
    df_h_v_o = run_eval("Heuristic", heur_agent, env, 50, "Optimized", opt_agent)
    df_h_v_r = run_eval("Heuristic", heur_agent, env, 50, "RL", rl_agent)
    df_o_v_r = run_eval("Optimized", opt_agent, env, 50, "RL", rl_agent)
    df_tourney = pd.concat([df_h_v_o, df_h_v_r, df_o_v_r])
    df_tourney.to_csv("results/tournament.csv", index=False)
    
    # Tourney plot
    wins = {
        "Heuristic": df_h_v_o["won"].sum() + df_h_v_r["won"].sum(),
        "Optimized": (50 - df_h_v_o["won"].sum()) + df_o_v_r["won"].sum(),
        "RL": (50 - df_h_v_r["won"].sum()) + (50 - df_o_v_r["won"].sum())
    }
    plt.figure(figsize=(8, 6))
    plt.bar(wins.keys(), wins.values(), color=['blue', 'orange', 'green'])
    plt.title("Liczba wygranych w Wielkim Turnieju (Cross-play)")
    plt.ylabel("Wygrane (max 100)")
    plt.savefig("results/tournament_wins.png")
    
    # Learning curves
    print("Generating learning curves")
    try:
        with open("agents/opt_history.json", "r") as f:
            opt_hist = json.load(f)
            plt.figure()
            for k, v in opt_hist.items():
                plt.plot(v, label=k)
            plt.legend()
            plt.title("Krzywa uczenia - Optymalizacja")
            plt.xlabel("Iteracje")
            plt.ylabel("Win Rate")
            plt.savefig("results/learning_curve_opt.png")
    except Exception as e:
        print("Could not load opt history", e)
        
    # RL checkpoints eval
    print("Eval early vs late RL checkpoints")
    checkpoints = ["1000000", "15000000", "30000000"]
    rl_progress = []
    from stable_baselines3 import PPO
    for cp in checkpoints:
        cp_path = f"agents/checkpoints/ppo_slimevolley_{cp}_steps.zip"
        if os.path.exists(cp_path):
            try:
                model = PPO.load(cp_path)
                agent = RLAgent(model=model)
                df = run_eval(f"RL_{cp}", agent, env, 20)
                rl_progress.append(df)
            except Exception as e:
                print(f"Error loading {cp_path}: {e}")
    if rl_progress:
        df_prog = pd.concat(rl_progress)
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df_prog, x="agent", y="our_points")
        plt.title("Postęp uczenia RL (Punkty vs Default)")
        plt.savefig("results/rl_progress.png")

if __name__ == "__main__":
    main()
