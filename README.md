⚙️ Setup & Installation

🔹 Prerequisites

Python 3.8–3.11 recommended

Git

🔹 Clone the Repository
git clone https://github.com/ankitaga00/drl-project
cd drl-project

🔹 Create & Activate a Virtual Environment
python -m venv venv


Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

🔹 Install Dependencies

pip install -r requirements.txt


🚦 Running the Project

🔹 1. Run the Fixed-Time Baseline

python src/run_fixed_baseline.py

🔹 2. Train the Independent RL Agents

python src/train_independent_rl.py

This trains four independent DQNs (one per intersection) and saves:

models/agent_0.pth … agent_3.pth

📌 Evaluate Independent RL

python src/eval_independent_rl.py

🔹 3. Train the CoLight-Style Cooperative Agent

python src/train_colight.py

This trains a CoLight-inspired model using adjacency awareness and stores:

models/trained_colight.pth

📌 Evaluate CoLight

python src/eval_colight.py

🔹 4. Visualize Traffic Flow Simulation

python src/visual_queues.py

You will see:

✔ signal switching (green/red)
✔ queues evolving
✔ vehicles queuing/moving

This is not SUMO-grade visualization — it is intentionally lightweight but interactive.

📊 Plot Comparisons

To compare all approaches:

python src/plot_comparison_curves.py
