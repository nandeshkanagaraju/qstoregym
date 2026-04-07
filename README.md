# Q-Store Gym 🛒

**Q-Store Gym** is an autonomous dark store operations simulator built for Reinforcement Learning (RL) agents. Developed for the OpenEnv Hackathon, this environment serves as a "Digital Twin" of a Q-Commerce warehouse where goods must be delivered in under 10 minutes.

## 🚀 Features
- **Dynamic Demand & Weather Events**: Real-time elements like storms or special events drastically change buyer urgency and rider availability.
- **Continuous Action Space**: Agents use dynamic pricing, inventory sourcing, and manual waste management.
- **Delayed Consequences**: A sophisticated Feedback Loop ensures the agent faces huge Waste Penalties if it over-stocks perishable goods.
- **Automated Curriculum Learning**: Progressive difficulty scaling.
- **OpenEnv Compliant**: Standard API structure (`step`, `reset`, `state`), standard config `openenv.yaml`, and full `pydantic` typing.

## ✨ Core API Definition

### ObservationSpace
The agent visually perceives the complete state of the dark store.
- **Inventory Health**: Stock levels, expiry timers, and cost basis.
- **Market Pulse**: Dynamic competitor pricing and a calculated `demand_index` per product.
- **Logistics**: The count of `available_riders`.
- **Environment**: Real-world noise `current_weather` & `special_event_active` flags.

### ActionSpace
- `pricing` (Dict[str, float]): A continuous map to control frontend retail prices.
- `sourcing` (Dict[str, int]): Volumes to order from suppliers (4-step lead time!).
- `waste_management` (Dict[str, int]): Map of quantities to discard manually to limit rot penalties.

### Rewards
The score algorithm returns `(Actual Net Profit / Max Potential Profit) - (Waste Ratio)`, where `Max Potential Profit` is the initial-inventory net-profit ceiling at the environment's max allowed markup and `Waste Ratio` is measured against inventory cost basis rather than profit. A `1.0` score means the agent matched that ceiling with zero waste.

## 🎯 Tasks & Expected Difficulty

Each task defines a concrete scenario that tests different aspects of operations management.

1. **The Night Shift** *(Difficulty: Easy)*
   A standard scenario with predictable customer volume and standard weather. Designed as a warmup task where baseline deterministic pricing easily hits positive margins.
2. **The Supplier Strike** *(Difficulty: Easy)*
   Sourcing capabilities are restricted, but demand remains steady. Agents must rely primarily on dynamic pricing.
3. **The Lunch Rush** *(Difficulty: Medium)*
   High peak concurrency for 2 hours surrounded by low-demand lulls. Requires aggressive inventory ramping and preemptive markdowns of perishable goods.
4. **The Weekend Blackout** *(Difficulty: Hard)*
   A severe storm limits rider availability to near 0, causing order backlogs, massive profit drops, and massive spoilage of short-expiry food items like Milk.
5. **The Strawberry Crisis** *(Difficulty: Hard)*
   The entire store is artificially flooded with highly perishable strawberries exactly 2 steps before rot. Requires flawless waste-management manual actions and fire-sale pricing.

## 📊 Baseline Scores

Our deterministic fallback baseline policy achieves the following exact reproducible `score` out of `1.0` (averaged over 3 environment seeds):

- **The Night Shift**: `0.150`
- **The Supplier Strike**: `0.150`
- **The Lunch Rush**: `0.067`
- **The Weekend Blackout**: `0.008`
- **The Strawberry Crisis**: `0.000`

## 🏃 Setup & Quickstart

### Local Setup
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### PPO Training
```bash
python train.py --device auto
```

To force GPU training when CUDA is available:
```bash
python train.py --device cuda
```

For AMD Radeon on Windows, install the DirectML backend first:
```bash
pip install torch-directml
python train.py --device dml
```
If you use `--device auto`, the script will try CUDA first, then DirectML, then CPU.

### Baseline Agent Inference (OpenAI `gpt-4o`)
Before execution, define the following variables as mandated by the spec:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="sk-proj-YOUR-KEY"
python inference.py
```

### Docker Usage
```bash
docker build -t q-store-gym .
docker run q-store-gym
```
