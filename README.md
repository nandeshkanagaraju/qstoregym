---
title: Q-Store Gym
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
---
# Q-Store Gym

**Q-Store Gym** is an autonomous dark store operations simulator developed for Reinforcement Learning (RL) agents. Designed specifically for the OpenEnv Hackathon, this environment serves as a rigorous "Digital Twin" of a modern Q-Commerce (Quick Commerce) fulfillment center where optimal performance requires processing and delivering perishable goods in restricted time limits.

## Core Features
- **Dynamic Demand & Stochastic Weather Modeling**: Incorporates real-time external conditions (e.g., severe weather systems, local public events) that organically perturb buyer urgency and delivery fleet availability.
- **Continuous Operations Action Space**: Agents exercise multifaceted control over operations, utilizing continuous dynamic pricing, inventory sourcing projections, and manual waste management logistics.
- **Delayed Consequence Architecture**: Employs a sophisticated feedback loop that significantly penalizes agents for poor long-term planning, specifically via extreme Waste Penalties resulting from over-indexing perishable stock.
- **Automated Curriculum Learning**: Features scalable difficulty progression to transition baseline models toward operational fluency.
- **OpenEnv Specification Compliance**: Built on a highly standardized API structure (`step`, `reset`, `state`), bundled with standard `openenv.yaml` configurations, and secured via comprehensive `pydantic` strict typing.

## API Specification

### ObservationSpace
The agent receives a heavily typed, complete multidimensional state matrix of the dark store:
- **Inventory Health**: Real-time stock volume computations, multi-step expiry degradation timers, and static cost basis points.
- **Market Pulse**: Real-time measurement of competitor pricing elasticity and an aggregated `demand_index` calculated per product constraint.
- **Logistics Capacity**: Live tally of `available_riders` within the active turn.
- **Environmental State**: Exposes variables including `current_weather` anomalies and `special_event_active` binary flags.

### ActionSpace
- `pricing` (Dict[str, float]): A continuous map setting the frontend retail markup multipliers to control demand.
- `sourcing` (Dict[str, int]): Volume purchase orders submitted to suppliers, requiring agents to account for multi-step fulfillment lead times.
- `waste_management` (Dict[str, int]): Volumes marked for manual write-off to aggressively mitigate compounded rot penalties on perishable inventory.

### Reward Function
The dense reward architecture returns `(Actual Net Profit / Max Potential Profit) - (Waste Ratio)`. `Max Potential Profit` represents the initial-inventory net-profit ceiling calculated at the environment's maximum allowed markup, while `Waste Ratio` penalizes agents based on initial component cost basis. A score of `1.0` signifies that the agent fulfilled peak potential revenue generation while effectively yielding zero physical waste.

## Evaluative Tasks & Difficulty

Each task defines an isolated, deterministic operational scenario that benchmarks specific dimensions of the agent's operations management capabilities:

1. **The Night Shift** *(Difficulty: Easy)*
   A baseline evaluation encompassing highly predictable customer volume and standard environmental states. Designed to establish fundamental markup execution margins.
2. **The Supplier Strike** *(Difficulty: Easy)*
   Severely restricts sourcing capabilities while maintaining steady demand. Forces agents to maximize margins entirely via algorithmic dynamic pricing.
3. **The Lunch Rush** *(Difficulty: Medium)*
   Simulates extreme peak transaction concurrency over fixed intervals followed by operational lulls. Demands highly preemptive inventory ramping and aggressive markdown liquidation strategies.
4. **The Weekend Blackout** *(Difficulty: Hard)*
   A severe environmental storm limits routing fleet availability strictly toward zero, resulting in cascading order backlogs, massive margin compression, and extensive spoilage of short-expiry food items.
5. **The Strawberry Crisis** *(Difficulty: Hard)*
   The dark store is artificially injected with surplus, highly perishable inventory precisely before rot decay boundaries. Agents must exhibit flawless manual waste management triage alongside fire-sale liquidation pricing models.

## Baseline Metrics

Our deterministic fallback baseline policy achieves the following exact reproducible `score` out of `1.0` (averaged over 3 environment seeds):

- **The Night Shift**: `0.150`
- **The Supplier Strike**: `0.150`
- **The Lunch Rush**: `0.067`
- **The Weekend Blackout**: `0.008`
- **The Strawberry Crisis**: `0.000`

## Implementation & Quickstart

### Environment Initialization
Ensure a Python 3.10+ execution environment is present.
```bash
pip install -r requirements.txt
```

### PPO Agent Training Procedures
```bash
python train.py --device auto
```

To enforce GPU utilization when CUDA is available:
```bash
python train.py --device cuda
```

For systems operating AMD Radeon hardware on Windows, install the DirectML backend initially:
```bash
pip install torch-directml
python train.py --device dml
```
*Note: Utilizing `--device auto` establishes precedence for CUDA, failing over to DirectML, and finally utilizing CPU execution.*

### Baseline Agent Inference (OpenAI `gpt-4o`)
Prior to script execution, specify the following variables as explicitly mandated by the OpenEnv compliance spec:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="sk-proj-YOUR-KEY"
python inference.py
```

### Containerized Execution (Docker)
```bash
docker build -t q-store-gym .
docker run q-store-gym
```
