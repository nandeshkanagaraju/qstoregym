from typing import Dict, Any

def get_task_config(task_name: str) -> Dict[str, Any]:
    if task_name == "The Night Shift":
        return {
            "initial_inventory": [
                {"product_id": "milk",  "quantity": 100, "cost_price": 2.0, "time_to_expiry_steps": 48},
                {"product_id": "bread", "quantity": 50,  "cost_price": 1.5, "time_to_expiry_steps": 24},
            ],
            # Per-product expiry used for ALL sourced batches (fixes hardcoded 100-step bug).
            "product_expiry_steps": {
                "milk":  48,
                "bread": 24,
            },
            "initial_riders": 5,
            "base_demand": 0.3,  # low demand
            "weather_prob": {"sunny": 0.8, "rainy": 0.1, "stormy": 0.0, "cloudy": 0.1},
            "special_event_prob": 0.0,
            "max_steps": 40,
        }
    elif task_name == "The Lunch Rush":
        return {
            "initial_inventory": [
                {"product_id": "milk",  "quantity": 150, "cost_price": 2.2, "time_to_expiry_steps": 24},
                {"product_id": "bread", "quantity": 200, "cost_price": 1.5, "time_to_expiry_steps": 12},
                {"product_id": "chips", "quantity": 100, "cost_price": 1.0, "time_to_expiry_steps": 100},
            ],
            "product_expiry_steps": {
                "milk":  24,
                "bread": 12,
                "chips": 100,
            },
            "initial_riders": 8,
            "base_demand": 1.5,  # high demand
            "weather_prob": {"sunny": 0.5, "rainy": 0.2, "stormy": 0.1, "cloudy": 0.2},
            "special_event_prob": 0.2,
            "max_steps": 40,
        }
    elif task_name == "The Strawberry Crisis":
        return {
            "initial_inventory": [
                {"product_id": "strawberries", "quantity": 150, "cost_price": 4.0, "time_to_expiry_steps": 10},
                {"product_id": "milk",         "quantity": 60,  "cost_price": 2.0, "time_to_expiry_steps": 36},
            ],
            "product_expiry_steps": {
                "strawberries": 10,
                "milk":         36,
            },
            # Demand is still elevated, but rider capacity and shorter expiry make greedy
            # high-markup policies leave real spoilage on the table.
            "initial_riders": 7,
            "base_demand": 1.4,
            "weather_prob": {"sunny": 0.1, "rainy": 0.4, "stormy": 0.5, "cloudy": 0.0},
            "special_event_prob": 0.7,
            "max_steps": 24,
        }
    elif task_name == "The Weekend Blackout":
        return {
            "initial_inventory": [
                {"product_id": "milk",  "quantity": 400, "cost_price": 2.0, "time_to_expiry_steps": 15},
                {"product_id": "bread", "quantity": 300, "cost_price": 1.5, "time_to_expiry_steps": 15},
                {"product_id": "chips", "quantity": 250, "cost_price": 1.0, "time_to_expiry_steps": 30},
            ],
            "product_expiry_steps": {
                "milk":  15,
                "bread": 15,
                "chips": 30,
            },
            "initial_riders": 8,
            "base_demand": 4.0,     # massive external demand
            "weather_prob": {"sunny": 0.0, "rainy": 1.0, "stormy": 0.0, "cloudy": 0.0},
            "special_event_prob": 1.0,  # guaranteed event
            "max_steps": 35,
        }
    elif task_name == "The Supplier Strike":
        return {
            "initial_inventory": [
                {"product_id": "chips", "quantity": 40, "cost_price": 1.0, "time_to_expiry_steps": 100},
                {"product_id": "bread", "quantity": 60, "cost_price": 1.5, "time_to_expiry_steps": 100},
            ],
            "product_expiry_steps": {
                "chips": 100,
                "bread": 100,
            },
            "initial_riders": 10,
            "base_demand": 1.0,
            "weather_prob": {"sunny": 1.0, "rainy": 0.0, "stormy": 0.0, "cloudy": 0.0},
            "special_event_prob": 0.0,
            "max_steps": 60,  # long depletion episode
        }
    else:
        raise ValueError(f"Unknown task: {task_name}")

AVAILABLE_TASKS = [
    "The Night Shift",
    "The Lunch Rush",
    "The Strawberry Crisis",
    "The Weekend Blackout",
    "The Supplier Strike",
]
