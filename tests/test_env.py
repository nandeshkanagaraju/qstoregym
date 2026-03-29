import unittest
from env import QStoreEnv
from models import ActionSpace

class TestQStoreEnv(unittest.TestCase):
    def setUp(self):
        self.env = QStoreEnv()
        self.obs = self.env.reset("The Night Shift")
        
    def test_initialization(self):
        self.assertEqual(self.env.current_step, 0)
        self.assertTrue(len(self.env.inventory) > 0)
        # Should have weather config setup
        self.assertTrue(self.env.current_weather in ['sunny', 'rainy', 'stormy', 'cloudy'])
        
    def test_step_pricing_sales(self):
        # Apply a simple deterministic action
        action = ActionSpace(
            pricing={"milk": 2.5, "bread": 2.0},
            sourcing={},
            waste_management={}
        )
        result = self.env.step(action)
        self.assertFalse(result.done)
        self.assertTrue(hasattr(result.observation, 'current_step'))
        self.assertEqual(result.observation.current_step, 1)
        # We should penalize some storage at least
        self.assertTrue(result.reward_breakdown.overhead_penalty >= 0)

    def test_sourcing_lead_time(self):
        action = ActionSpace(
            pricing={},
            sourcing={"milk": 50},
            waste_management={}
        )
        self.env.step(action)
        # Should have a pending order looking forward
        pending = [o for o in self.env.pending_orders if o[1] == "milk"]
        self.assertTrue(len(pending) == 1)
        
    def test_waste_management(self):
        initial_waste = self.env.total_waste_value
        action = ActionSpace(
            pricing={},
            sourcing={},
            waste_management={"milk": 10}
        )
        result = self.env.step(action)
        # Ensure waste penalty is registered manually
        self.assertTrue(result.reward_breakdown.waste_penalty > 0)
        self.assertTrue(self.env.total_waste_value > initial_waste)

if __name__ == '__main__':
    unittest.main()
