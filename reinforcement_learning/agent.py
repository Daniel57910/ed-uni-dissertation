USER_INDEX = 1
SESSION_INDEX = 2
TASK_INDEX = 3
import numpy as np


class Constant20Agent:
    def __init__(self):
        self.action_space = [0, 1]
        self.name = 'Constant20Agent'
        
    def action(self, step):
        return self.action_space[0]if step[:, TASK_INDEX] < 20 else self.action_space[1]

class Random20Agent:
    def __init__(self):
        self.action_space = [0, 1]
        self.name = 'Random20Agent'
        
    def action(self, step):
        likelihood_action = np.random.uniform(0, 1)
        return self.action_space[0] if likelihood_action > 0.2 else self.action_space[1]
        
AGENT_META = {
    'constant_20': Constant20Agent,
    'random_20': Random20Agent,
}