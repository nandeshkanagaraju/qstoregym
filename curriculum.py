from typing import List
from env import QStoreEnv
from models import ObservationSpace, ActionSpace, StepResult
from tasks import AVAILABLE_TASKS

class CurriculumWrapper:
    def __init__(self, env: QStoreEnv, promotion_threshold: float = 0.8, consecutive_episodes_required: int = 3):
        self.env = env
        self.promotion_threshold = promotion_threshold
        self.consecutive_episodes_required = consecutive_episodes_required
        
        self.current_task_idx = 0
        self.performance_history: List[float] = []
        
    def reset(self) -> ObservationSpace:
        task_name = AVAILABLE_TASKS[self.current_task_idx]
        return self.env.reset(task_name=task_name)
        
    def step(self, action: ActionSpace) -> StepResult:
        result = self.env.step(action)
        if result.done:
            self.performance_history.append(result.score)
            self._check_promotion()
        return result
        
    def _check_promotion(self):
        if len(self.performance_history) >= self.consecutive_episodes_required:
            recent_scores = self.performance_history[-self.consecutive_episodes_required:]
            if all(s >= self.promotion_threshold for s in recent_scores):
                if self.current_task_idx < len(AVAILABLE_TASKS) - 1:
                    self.current_task_idx += 1
                    self.performance_history = [] # reset history for new task
                    print(f"*** Curriculum Promotion! Advanced to {AVAILABLE_TASKS[self.current_task_idx]} ***")
                
    def state(self) -> ObservationSpace:
        return self.env.state()
