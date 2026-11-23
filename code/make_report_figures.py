"""Build the report's figures and tables from the experiment outputs.

Reads ../ARC_RL_Project/results/evaluation_results.csv (one row per version x seed x held-out game) and
the per-run Monitor training logs (*.monitor.csv), and writes PDF figures into ./figures/ plus the
summary numbers the report quotes into ./figures/summary.txt. Same data and logic as the notebook's
Task 8 cell, with the review fixes (smoke-test rows excluded, spread computed across seeds).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPORT_DIR.parent / "ARC_RL_Project" / "results"
FIGURES_DIR = REPORT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

VERSION_ORDER = ["dqn_v0_baseline", "dqn_v1_memory", "dqn_v2_memory_novelty", "dqn_v2_novelty_only",
                 "ppo_v0_baseline", "ppo_v1_memory", "ppo_v2_memory_novelty", "ppo_v2_novelty_only"]
SHORT_NAME = {"dqn_v0_baseline": "v0", "dqn_v1_memory": "v1 mem", "dqn_v2_memory_novelty": "v2 mem+nov",
              "dqn_v2_novelty_only": "nov only", "ppo_v0_baseline": "v0", "ppo_v1_memory": "v1 mem",
              "ppo_v2_memory_novelty": "v2 mem+nov", "ppo_v2_novelty_only": "nov only", "random_v0": "random"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})
summary_lines = []

# ------------------------------------------------------------------ evaluation results (held-out games)
evaluation = pd.read_csv(RESULTS_DIR / "evaluation_results.csv")
evaluation = evaluation[~evaluation["version"].str.startswith("selfcheck")]
evaluation["game"] = evaluation["game_id"].str[:4]

per_seed = evaluation.groupby(["version", "seed"]).agg(levels=("levels_completed", "sum"),
                                                       actions=("actions_used", "mean"),
                                                       invalid=("invalid_actions", "mean"))
spread = per_seed.groupby("version").agg(["mean", "std"]).round(1)
summary_lines += ["HELD-OUT, per version: mean and std ACROSS SEEDS of per-seed totals/means", spread.to_string(), ""]

invalid_by_game = evaluation.pivot_table(index="version", columns="game", values="invalid_actions", aggfunc="mean").round(0)
actions_by_game = evaluation.pivot_table(index="version", columns="game", values="actions_used", aggfunc="mean").round(0)
summary_lines += ["HELD-OUT invalid actions by game (mean over seeds)", invalid_by_game.to_string(), "",
                  "HELD-OUT actions used by game (mean over seeds)", actions_by_game.to_string(), ""]

# Figure: wasted (invalid) actions per held-out game, per version, seeds as dots
figure, axis = plt.subplots(figsize=(6.5, 2.4))
versions = ["random_v0"] + VERSION_ORDER
positions = np.arange(len(versions))
per_version_seed_invalid = evaluation.groupby(["version", "seed"])["invalid_actions"].mean()
means = [per_version_seed_invalid.loc[v].mean() for v in versions]
colours = ["0.6"] + ["tab:blue"] * 4 + ["tab:orange"] * 4
axis.bar(positions, means, color=colours, alpha=0.75)
for position, version in zip(positions, versions):
    seed_values = per_version_seed_invalid.loc[version].values
    axis.scatter(np.full(len(seed_values), position), seed_values, color="black", s=8, zorder=3)
axis.set_xticks(positions)
axis.set_xticklabels([("DQN " if v.startswith("dqn") else "PPO " if v.startswith("ppo") else "") + SHORT_NAME[v]
                      for v in versions], rotation=30, ha="right")
axis.set_ylabel("invalid actions per game")
axis.set_title("Held-out evaluation: actions spent on unavailable buttons (bars: mean over seeds; dots: seeds)")
figure.tight_layout()
figure.savefig(FIGURES_DIR / "invalid_actions.pdf"); figure.savefig(FIGURES_DIR / "invalid_actions.png", dpi=110)
plt.close(figure)

# ------------------------------------------------------------------ training logs (Monitor files)
def load_monitor(version, seed):
    log = pd.read_csv(RESULTS_DIR / f"{version}_seed{seed}.monitor.csv", skiprows=1)
    log["steps"] = log["l"].cumsum()
    return log

training_rows = []
for version in VERSION_ORDER:
    for seed in (0, 1, 2):
        log = load_monitor(version, seed)
        shaped = "novelty" in version
        training_rows.append({"version": version, "seed": seed, "episodes": len(log),
                              "mean_episode_length": round(log["l"].mean()),
                              "levels_in_training": None if shaped else int(log["r"].sum()),
                              "shaped_return_total": round(log["r"].sum(), 1) if shaped else None,
                              "train_minutes": round(log["t"].iloc[-1] / 60, 1)})
training = pd.DataFrame(training_rows)
summary_lines += ["TRAINING (Monitor logs). levels_in_training = true reward sum (non-novelty runs only)",
                  training.to_string(), "",
                  "Training minutes per version (mean over seeds)",
                  training.groupby("version")["train_minutes"].mean().round(1).to_string(),
                  f"Total training minutes, all 24 runs: {training['train_minutes'].sum():.0f}", ""]

# Figure: cumulative levels completed during training (true reward), non-novelty runs
figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True)
for axis, algorithm, colour_map in [(axes[0], "dqn", {"dqn_v0_baseline": "tab:blue", "dqn_v1_memory": "tab:cyan"}),
                                    (axes[1], "ppo", {"ppo_v0_baseline": "tab:orange", "ppo_v1_memory": "tab:red"})]:
    for version, colour in colour_map.items():
        for seed in (0, 1, 2):
            log = load_monitor(version, seed)
            axis.step(log["steps"], log["r"].cumsum(), where="post", color=colour, alpha=0.8,
                      label=SHORT_NAME[version] if seed == 0 else None)
    axis.set_title(f"{algorithm.upper()}: levels completed during training")
    axis.set_xlabel("training steps")
    axis.legend(loc="upper left")
axes[0].set_ylabel("cumulative levels (one line per seed)")
figure.tight_layout()
figure.savefig(FIGURES_DIR / "training_levels.pdf"); figure.savefig(FIGURES_DIR / "training_levels.png", dpi=110)
plt.close(figure)

# Figure: episode length during training, all versions (shows behavioural change incl. novelty runs)
figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True)
for axis, algorithm in [(axes[0], "dqn"), (axes[1], "ppo")]:
    for version, colour in zip([v for v in VERSION_ORDER if v.startswith(algorithm)],
                               ["tab:blue", "tab:cyan", "tab:green", "tab:purple"]):
        lengths = [load_monitor(version, seed) for seed in (0, 1, 2)]
        grid = np.linspace(0, 50_000, 100)
        curves = [np.interp(grid, log["steps"], log["l"].rolling(20, min_periods=1).mean()) for log in lengths]
        mean_curve = np.mean(curves, axis=0)
        axis.plot(grid, mean_curve, color=colour, label=SHORT_NAME[version])
        axis.fill_between(grid, np.min(curves, axis=0), np.max(curves, axis=0), color=colour, alpha=0.15)
    axis.set_title(f"{algorithm.upper()}: episode length while training")
    axis.set_xlabel("training steps")
    axis.legend(loc="upper right")
axes[0].set_ylabel("steps per episode (rolling 20)")
figure.tight_layout()
figure.savefig(FIGURES_DIR / "episode_length.pdf"); figure.savefig(FIGURES_DIR / "episode_length.png", dpi=110)
plt.close(figure)

(FIGURES_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
print("\n".join(summary_lines))
print("Figures written to", FIGURES_DIR)
