import re
import matplotlib.pyplot as plt

def extract_rewards(filename):
    """
    Reads a log file and extracts all numeric reward values.
    Supports logs like:
    'Episode 10 -> Total reward -3056.3'
    """
    rewards = []
    try:
        with open(filename, "r") as f:
            for line in f:
                match = re.search(r"reward\s*[-]?\d+\.?\d*", line.lower())
                if match:
                    # extract last number in the match
                    value = float(match.group(0).split()[-1])
                    rewards.append(value)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
    return rewards


# ====== Load Logs ======
log_fixed = extract_rewards("logs/train_fixed_baseline.log")   # adjust if named differently
log_independent = extract_rewards("logs/train_independent_rl.log")
log_colight = extract_rewards("logs/train_colight.log")

# Filter out empty ones
labels = []
curves = []

if log_fixed:
    labels.append("Fixed-Time")
    curves.append(log_fixed)

if log_independent:
    labels.append("Independent RL")
    curves.append(log_independent)

if log_colight:
    labels.append("CoLight RL")
    curves.append(log_colight)


# ====== Plot ======
plt.figure(figsize=(10, 5))

for rewards, label in zip(curves, labels):
    plt.plot(rewards, label=label, linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Performance Comparison: Fixed vs Independent RL vs CoLight")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("compare_performance.png")
plt.show()

print("✔ Plot saved as compare_performance.png")
