# Q-Store Gym 🛒

**Q-Store Gym** is an autonomous dark store operations simulator built for Reinforcement Learning (RL) agents. Developed for the OpenEnv Hackathon, this environment serves as a "Digital Twin" of a Q-Commerce warehouse where goods must be delivered in under 10 minutes.

## 🚀 Features
- **Dynamic Demand & Weather Events**: Real-time elements like storms or special events drastically change buyer urgency and rider availability.
- **Continuous Action Space**: Agents use dynamic pricing, inventory sourcing, and manual waste management.
- **Delayed Consequences**: A sophisticated Feedback Loop ensures the agent faces huge Waste Penalties if it over-stocks perishable goods.
- **Automated Curriculum Learning**: Progressive difficulty (Night Shift -> Lunch Rush -> Strawberry Crisis).
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
The score algorithm returns `(Actual Net Profit / Max Potential Profit) - (Waste Ratio)`. Maximizing sales while keeping absolute 0 waste equals a `1.0` score.

## 🏃 Setup & Quickstart

### Local Setup
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### Baseline Agent Inference (OpenAI `gpt-4o`)
```bash
export OPENAI_API_KEY="your-key"
python inference.py
```

### Docker Usage
```bash
docker build -t q-store-gym .
docker run q-store-gym
```
