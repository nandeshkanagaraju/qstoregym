# 🛒 Q-Store Gym: Continuous Dark Store Simulation

> **Q-Store Gym** is an autonomous dark store operations simulator. Built with compliance to the OpenEnv standard, this environment acts as a "Digital Twin" for an ultra-fast Q-commerce warehouse (like Getir or Gopuff) where perishables must be sold and delivered under stringent logistics constraints.

---

## 🚀 Features

- **Continuous Simulation Mechanics**: Real-time modeling of dynamic competitor pricing, variable market demand, multi-step sourcing lead times, and exact shelf-life rotting clocks.
- **Dynamic Logistics**: Rider capacities scale and diminish fluidly based on sudden weather events (Storms, Rain) and special marketing rushes.
- **Deep Penalty Signals**: Actions carry massive ripple effects. Lowering prices perfectly triggers high demand, but if you lack riders to deliver them, you get crushed by Trust Penalties. Over-ordering incurs massive Rot and Storage penalties.
- **Dual Inference Engine**: Evaluate using **Zero-Shot LLMs** (like ChatGPT-4o) for high-level strategy checks, or train hyper-optimized mathematical Local Agents using **Proximal Policy Optimization (PPO)**. 

---

## 🧬 Project Architecture
What was initially a standard logic environment has been heavily upgraded to support true localized Reinforcement Learning. 

- `env.py`: The core simulator. It calculates continuous sales, dynamic weather algorithms, rotting logic, and returns deep penalty/reward step tracking arrays. It now includes localized console printing logic for highly readable simulation tracing.
- `gym_wrapper.py`: The mathematics adapter. Because the core `env.py` operates on heavily structured Pydantic models (like `ActionSpace`), the `QStoreGymWrapper` flawlessly translates generic neural-network output arrays `[-1.0, 1.0]` back up into bounded, valid economic decisions (pricing `$0.50-$4.50`, restricted sourcing capacities, etc.) so gradient explosions don't occur.
- `train.py`: Initializes the localized Neural Network (PPO from `stable-baselines3`) and automatically parallel-trains an exact replica agent specialized for each 3 of the environment challenges.
- `inference.py`: An intelligent inference wrapper that uses `argparse` to cleanly load your preferred brain (GPT vs Local Neural Network).

---

## 🏃 Setup & Quickstart

### 1. Local Setup
We recommend setting up a virtual environment. Ensure you have Python 3.10+ installed.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Evaluator Inference (Native Deep Learning)
The repository comes out-of-the-box supporting `stable-baselines3`. Assuming you have run the training script, the inference engine defaults securely to the trained agents.
```bash
python inference.py
```
> **Output:** It will automatically cycle through the 3 difficulty scenarios (Night Shift, Lunch Rush, Strawberry Crisis), load the specifically trained mathematical Weights for each (`ppo_The_Night_Shift.zip`), and stream the internal step-by-step PnL (Profit & Loss) logs directly to your terminal.

### 3. Evaluator Inference (Zero-Shot AI)
If you wish to test Big Tech AI Models against your locally-trained mathematical ones, you can supply your API key natively to `.env` and force ChatGPT execution:
```bash
# 1. Open the `.env` file that exists in the root directory and add your key:
OPENAI_API_KEY="sk-proj-YourKey"

# 2. Run with the override flag:
python inference.py --use-gpt
```

### 4. Continuous Retraining
If you want to bake the Neural Networks for longer, you can invoke the trainer. It's configured to train 100,000 algorithmic trial-and-error steps for every unique scenario:
```bash
python train.py
```

---

## ✨ Core Mechanics Definition

### Difficulty Scenarios
1. **The Night Shift**: Baseline training. Long expiry times, high delivery capacities. Easily mastered by an RL algorithm.
2. **The Lunch Rush**: Chaos variables. Multiple goods rotting at once, surging demand mechanics. The agent must flawlessly price out competitors to clear the shelves without over-booking its riders.
3. **The Strawberry Crisis**: Disaster recovery. Sudden flash-rot mechanics (strawberries die in 12 steps) paired with a negative rider deficit. Extremely difficult.

### The Reward Optimization Problem
The Neural Network's singular goal at every snapshot is maximizing positive Reward delta.
`score = max(0.0, profit_ratio - waste_ratio)`
The environment punishes selling at a negative margin, punishes holding too much inventory, explicitly punishes rot, and exponentially punishes taking on orders that physically cannot be delivered. Obtaining a 1.0 score means literal economic perfection.
