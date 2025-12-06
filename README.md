🚦 CoLight-Inspired Cooperative Deep RL for Traffic Signal Control
College Station Road Network Simulation (4-Intersection Prototype)

📌 Overview

This repository contains a CoLight-inspired multi-intersection traffic signal control project developed as part of a graduate Deep Reinforcement Learning coursework final project.

The goal is to examine whether cooperative RL can outperform fixed-time and independent RL controllers when managing traffic flow in a simplified model of College Station, Texas.

The implementation:

✔ Extracts a real subnetwork topology via OpenStreetMap
✔ Builds a custom stochastic traffic environment
✔ Implements three controllers:
    ▫ Fixed-time baseline
    ▫ Independent RL (per-intersection learning)
    ▫ CoLight-inspired cooperative RL
✔ Visualizes signal switching & vehicle movements
✔ Produces comparable performance metrics

This repository satisfies the course requirements: sequential decision making, experiment design, evaluation baselines, visualization, and written report.

📂 Project Structure

📦 traffic-colight-cs/
│
├── multi_intersection_env.py       # Core traffic environment

├── single_intersection_env.py      # Local intersection dynamics

│
├── colight_agent.py                # Cooperative RL agent (CoLight-inspired)


├── train_colight.py                # Training script


├── eval_colight.py                 # Evaluation script

│
├── train_independent_rl.py         # Independent RL baseline

├── eval_independent_rl.py          # Independent baseline evaluation

│
├── run_fixed_baseline.py           # Fixed-time controller baseline

│
├── visual_sim.py                   # Pygame traffic visualization

│
├── logger.py                       # Experiment logging utility

│
├── results/                        # Output logs / reward curves

│
└── README.md                       # This file


📌 Running Experiments

✔ Train Cooperative CoLight-Inspired Model

python train_colight.py

This:

Runs 100 episodes

Saves model as trained_colight.pth

Logs results under /results/

✔ Evaluate Cooperative Model

python eval_colight.py

Outputs average reward over test runs.

✔ Run Independent RL Baseline

Training:

python train_independent_rl.py


Evaluation:

python eval_independent_rl.py


✔ Produces baseline performance for comparison.

✔ Run Fixed-Time Baseline
python run_fixed_baseline.py

🎥 Visualization

To view intersection dynamics:

python visual_sim.py


This launches a Pygame window displaying:

🟥/🟩 signal switching
🚗 vehicle queues forming and moving
🔁 sequential evolution over time

NOTE: The visualization is simplified and intended to illustrate qualitative behavior (queue growth/shrink, signal influence), not a physics-accurate traffic simulator.
