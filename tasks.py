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
                {"product_id": "strawberries", "quantity": 150, "cost_price": 4.0, "time_to_expiry_steps": 10}, # Reduced rottable stock to beatable sizes
                {"product_id": "milk", "quantity": 100, "cost_price": 2.0, "time_to_expiry_steps": 48}
            ],
            "initial_riders": 5, # Starts with 5 riders (20 cap/step -> 200 possible in 10 steps!)
            "base_demand": 2.5, # Sudden influx
            "weather_prob": {"sunny": 0.1, "rainy": 0.4, "stormy": 0.5, "cloudy": 0.0}, 
            "special_event_prob": 0.8,
            "max_steps": 24
        }
    elif task_name == "The Weekend Blackout":
        return {
            "initial_inventory": [
                {"product_id": "milk", "quantity": 400, "cost_price": 2.0, "time_to_expiry_steps": 15},
                {"product_id": "bread", "quantity": 300, "cost_price": 1.5, "time_to_expiry_steps": 15},
                {"product_id": "chips", "quantity": 250, "cost_price": 1.0, "time_to_expiry_steps": 30}
            ],
            "initial_riders": 8, # Enough initial riders to struggle but survive
            "base_demand": 4.0, # Massive external demand
            "weather_prob": {"sunny": 0.0, "rainy": 1.0, "stormy": 0.0, "cloudy": 0.0}, # Heavy rain instead of storm to preserve riders
            "special_event_prob": 1.0, # Guaranteed event
            "max_steps": 35 
        }
    elif task_name == "The Supplier Strike":
        return {
            "initial_inventory": [
                {"product_id": "chips", "quantity": 40, "cost_price": 1.0, "time_to_expiry_steps": 100},
                {"product_id": "bread", "quantity": 60, "cost_price": 1.5, "time_to_expiry_steps": 100}
            ],
            "initial_riders": 10, # Lots of riders available
            "base_demand": 1.0, 
            "weather_prob": {"sunny": 1.0, "rainy": 0.0, "stormy": 0.0, "cloudy": 0.0},
            "special_event_prob": 0.0,
            "max_steps": 60 # Extremely long depletion episode
        }
    else:
        raise ValueError(f"Unknown task: {task_name}")

AVAILABLE_TASKS = ["The Night Shift", "The Lunch Rush", "The Strawberry Crisis", "The Weekend Blackout", "The Supplier Strike"]
