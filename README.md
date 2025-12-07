# COLIGHT-LITE

**Team Members:**
- Ankita Girish Aswathanarayana
- Prajnadipta Kundu

## Overview

**CoLight-Lite** is a lightweight multi-agent reinforcement learning framework designed to explore cooperative traffic signal control across connected intersections. Inspired by the original CoLight architecture (Wei et al., 2019), this project implements a simplified but functional version of graph-based coordination using adjacency-aware Q-networks rather than full attention mechanisms.

The system simulates a four-intersection region in College Station, TX, enabling comparative evaluation between:

* **Fixed-time control**
* **Independent RL controllers (DQN)**
* **Cooperative RL using network structure (CoLight-Lite)**

This project was developed as a **Deep Reinforcement Learning Project**.

## Key Features

* **Graph-Aware Cooperative RL:**
A simplified CoLight-style coordination mechanism where each agent incorporates neighboring intersection states via adjacency-conditioned Q-values.

* **Custom Multi-Intersection Simulator:**
A lightweight stochastic traffic simulator built specifically for experimentation, supporting queues, arrival processes, and phase switching.

* **Independent vs. Cooperative RL Baselines:**
Side-by-side training pipelines for independent DQNs and cooperative agents, enabling fair comparison.

* **Explainable Metrics:**
The system reports queue lengths, travel delays, and episode-level rewards for transparent evaluation.

* **Visualization Tools:**
Pygame-based animations display intersection states, vehicle queues, and real-time signal switching.

## Project Architecture

CoLight-Lite follows a modular architecture to support clarity, extensibility, and experimentation:

Environment Simulation:
Custom Python environments (single_intersection_env.py, multi_intersection_env.py) implement queue dynamics, stochastic arrivals, and signal logic.

Agent Models:

Independent DQN agents (dqn_agent.py)

CoLight-Lite cooperative agent (colight_agent.py, colight_qnet.py)

Training & Evaluation:
Scripts for running fixed baselines, training RL models, and evaluating learned policies:

run_fixed_baseline.py

train_independent_rl.py, train_colight.py

eval_independent_rl.py, eval_colight.py

Prompt Assembly (Cooperative Inputs):
The cooperative agent constructs a joint observation vector using graph adjacency, allowing each intersection to embed its neighbors' queue states.

Inference Loop:
During runtime, each intersection selects actions, updates Q-values, and synchronizes reward statistics to enable consistent evaluation across baselines.

Visualization:
Real-time queue animations are rendered using:

visual_sim.py

visual_queues.py
  
Setup & Installation

This guide walks you through installing and running CoLight-Lite, a simplified multi-agent reinforcement learning framework for traffic signal control.

🔹 Prerequisites

Make sure you have the following installed:

Python 3.8–3.11 (recommended)

Git

(Optional) A GPU machine if you want faster RL training

🔹 Clone the Repository
git clone https://github.com/ankitaga00/drl-project.git
cd drl-project

🔹 Create & Activate a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

Mac / Linux
python3 -m venv venv
source venv/bin/activate

🔹 Install Dependencies
pip install -r requirements.txt

🚦 Running the Project

The project includes three traffic control approaches:

Fixed-Time Controller

Independent DQN Agents

CoLight-Lite Cooperative RL

🔹 1. Run the Fixed-Time Baseline
python src/run_fixed_baseline.py


This executes the non-learning baseline using simple periodic timing.

🔹 2. Train the Independent RL Agents
python src/train_independent_rl.py


This trains four independent DQN agents (one per intersection).

After training, the models are saved in:

models/
 ├── agent_0.pth
 ├── agent_1.pth
 ├── agent_2.pth
 └── agent_3.pth

📌 Evaluate the Independent RL Performance
python src/eval_independent_rl.py

🔹 3. Train the CoLight-Lite Cooperative Agent
python src/train_colight.py


This trains a graph-aware cooperative RL controller inspired by CoLight.

Model will be saved as:

models/trained_colight.pth

📌 Evaluate CoLight Performance
python src/eval_colight.py

🔹 4. Visualize the Traffic Flow Simulation
python src/visual_queues.py


This displays a lightweight 2D simulation showing:

✔ Signal switching (red → green)

✔ Queue lengths evolving over time

✔ Vehicle movements

Note: This visualization is intentionally simple and designed for clarity—not SUMO-level fidelity.

📊 Comparing All Approaches

To generate comparative plots (queue length, reward curves, etc.):

python src/plot_comparison_curves.py


This outputs PNG graphs summarizing:

Fixed-time baseline

Independent DQN agents

CoLight-Lite cooperative RL
