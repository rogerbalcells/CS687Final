import numpy as np

class NStepSARSAAgent:
    def __init__(self, model, action_cnt, gamma, n_step, epsilon):
        self.model = model 
        self.action_space_size = action_cnt
        self.gamma = gamma
        self.n_step = n_step
        self.epsilon = epsilon
        self.buffer = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)
        
        return self.model.predict(state)

    def update(self, t, T_terminal):
        tau = t - self.n_step + 1
        if tau < 0: return

        G = 0.0
        end_idx = min(tau + self.n_step, T_terminal)
        
        for i in range(tau, end_idx):
            reward = self.buffer[i][2]
            G += (self.gamma ** (i - tau)) * reward

        if tau + self.n_step < T_terminal:
            s_next, a_next, _ = self.buffer[tau + self.n_step]
            q_next_val = self.model.get_value(s_next, a_next)
            G += (self.gamma ** self.n_step) * q_next_val

        s_tau, a_tau, _ = self.buffer[tau]
        self.model.update(s_tau, a_tau, G)

    def save_transition(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def clear_buffer(self):
        self.buffer = []