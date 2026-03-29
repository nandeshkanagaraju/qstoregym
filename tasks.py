from typing import Dict, Any

def get_task_config(task_name: str) -> Dict[str, Any]:
    if task_name == "The Night Shift":
        return {
            "initial_inventory": [
                {"product_id": "milk", "quantity": 100, "cost_price": 2.0, "time_to_expiry_steps": 48},
                {"product_id": "bread", "quantity": 50, "cost_price": 1.5, "time_to_expiry_steps": 24}
            ],
            "initial_riders": 5,
            "base_demand": 0.3, # low demand
            "weather_prob": {"sunny": 0.8, "rainy": 0.1, "stormy": 0.0, "cloudy": 0.1},
            "special_event_prob": 0.0,
            "max_steps": 40
        }
    elif task_name == "The Lunch Rush":
        return {
            "initial_inventory": [
                {"product_id": "milk", "quantity": 150, "cost_price": 2.2, "time_to_expiry_steps": 24},
                {"product_id": "bread", "quantity": 200, "cost_price": 1.5, "time_to_expiry_steps": 12},
                {"product_id": "chips", "quantity": 100, "cost_price": 1.0, "time_to_expiry_steps": 100}
            ],
            "initial_riders": 8,
            "base_demand": 1.5, # high demand
            "weather_prob": {"sunny": 0.5, "rainy": 0.2, "stormy": 0.1, "cloudy": 0.2},
            "special_event_prob": 0.2,
            "max_steps": 40
        }
    elif task_name == "The Strawberry Crisis":
        return {
            "initial_inventory": [
                {"product_id": "strawberries", "quantity": 200, "cost_price": 4.0, "time_to_expiry_steps": 12}, # Flash expiry event: 12 steps (3 hours)
                {"product_id": "milk", "quantity": 100, "cost_price": 2.0, "time_to_expiry_steps": 48}
            ],
            "initial_riders": 2, # Severe rider shortage
            "base_demand": 2.0, # High sudden influx
            "weather_prob": {"sunny": 0.1, "rainy": 0.4, "stormy": 0.5, "cloudy": 0.0}, # Bad weather
            "special_event_prob": 0.5,
            "max_steps": 24 # crisis is shorter
        }
    else:
        raise ValueError(f"Unknown task: {task_name}")

AVAILABLE_TASKS = ["The Night Shift", "The Lunch Rush", "The Strawberry Crisis"]
