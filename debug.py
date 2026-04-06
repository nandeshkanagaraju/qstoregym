import traceback
from env import QStoreEnv
from models import ActionSpace

try:
    env = QStoreEnv()
    env.reset("The Strawberry Crisis")

    done = False
    while not done:
        action = ActionSpace(
            pricing={"strawberries": 1.3, "milk": 1.3}, 
            sourcing={}, 
            waste_management={}
        )
        result = env.step(action, verbose=True)
        done = result.done

    print(f"Final Score: {result.score}")
except Exception as e:
    traceback.print_exc()
