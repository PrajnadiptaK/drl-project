import os
import re
import matplotlib.pyplot as plt

LOG_DIR = "logs"

def get_latest_log(prefix):
    """Find most recent log matching given prefix."""
    matches = [f for f in os.listdir(LOG_DIR) if f.startswith(prefix)]
    if not matches:
        return None
    return os.path.join(LOG_DIR, sorted(matches)[-1])  # newest log


def extract_rewards(filepath):
    rewards = []
    if not filepath or not os.path.exists(filepath):
        return rewards
    with open(filepath, "r") as f:
        for line in f:
            if "Total reward" in line:
                value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])
                rewards.append(value)
    return rewards


# ================= TRAINING PLOTS =====================

train_logs = {
    "Fixed-time": get_latest_log("fixed_baseline"),
    "Independent RL": get_latest_log("train_independent_rl"),
    "CoLight": get_latest_log("train_colight"),
}

train_data = {k: extract_rewards(v) for k, v in train_logs.items()}

if any(train_data.values()):
    plt.figure(figsize=(10, 5))
    for method, rewards in train_data.items():
        if rewards:
            plt.plot(rewards, label=method)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Comparison")
    plt.legend()
    plt.grid()
    plt.savefig("train_plot.png", dpi=150)
    print("Saved train_plot.png")
else:
    print("No training logs detected — skipping training plot.")


# ================= EVALUATION PLOTS =====================

eval_logs = {
    "Fixed-time": get_latest_log("eval_fixed_time"),
    "Independent RL": get_latest_log("eval_independent_rl"),
    "CoLight": get_latest_log("eval_colight"),
}

eval_data = {k: extract_rewards(v) for k, v in eval_logs.items()}

if any(eval_data.values()):
    plt.figure(figsize=(10, 5))
    for method, rewards in eval_data.items():
        if rewards:
            plt.plot(rewards, label=method)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Evaluation Comparison")
    plt.legend()
    plt.grid()
    plt.savefig("eval_plot.png", dpi=150)
    print("Saved eval_plot.png")
else:
    print("No evaluation logs detected — skipping evaluation plot.")
