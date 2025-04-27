import gym
from gym import spaces
import numpy as np
import pandas as pd

class FraudTransactionEnv(gym.Env):
    def __init__(self, csv_path):
        super(FraudTransactionEnv, self).__init__()

        self.data = pd.read_csv(csv_path)
        self.features = self.data.drop(columns=['Class']).values
        self.labels = self.data['Class'].values

        self.n_samples = len(self.data)
        self.current_index = 0

        # Définir espace d'observation et d'action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: Accepter, 1: Bloquer

    def reset(self):
        self.current_index = np.random.randint(0, self.n_samples)
        state = self.features[self.current_index]
        return state

    def step(self, action):
        label = self.labels[self.current_index]

        if action == 1 and label == 1:
            reward = 1.0 
        elif action == 0 and label == 0:
            reward = 1.0 
        else:
            reward = -10.0 

        self.current_index = np.random.randint(0, self.n_samples)
        next_state = self.features[self.current_index]

        done = False 
        info = {}

        return next_state, reward, done, info
