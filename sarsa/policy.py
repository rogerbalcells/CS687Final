import numpy as np
import torch
import torch.nn as nn

MIN_POSITION = -1.2
MAX_POSITION = 0.5
MIN_VELOCITY = -0.07
MAX_VELOCITY = 0.07
TIMEOUT      = 1000

class MountainCarNet(nn.Module):
    def __init__(self, layers, alpha):
        super().__init__()
        
        self.alpha = alpha

        net_layers = []
        n_inputs = 2 # Number of inputs  is 2 because the state is s=[x,v])
        n_outputs = 3 # Number of outputs is 3 because the network outputs one score value 
        
        last = n_inputs
        for h in layers:
            net_layers.append(nn.Linear(last, h))
            net_layers.append(nn.Tanh())
            last = h
        net_layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*net_layers)
        
        # Precompute constants needed to later perform [-1, 1] normalization when given an unnormalized state s=[x,v] as input
        self.pos_mid  = 0.5 * (MIN_POSITION + MAX_POSITION)
        self.pos_half = 0.5 * (MAX_POSITION - MIN_POSITION)
        self.vel_mid  = 0.5 * (MIN_VELOCITY + MAX_VELOCITY)
        self.vel_half = 0.5 * (MAX_VELOCITY - MIN_VELOCITY)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[..., 0]
        vel = x[..., 1]
        pos_n = (pos - self.pos_mid) / self.pos_half
        vel_n = (vel - self.vel_mid) / self.vel_half
        return torch.stack([pos_n, vel_n], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self._normalize(x)
        return self.net(x_n)

    def _to_tensor(self, state_np):
        return torch.tensor(state_np, dtype=torch.float32)

    def predict(self, state_np):
        with torch.no_grad():
            t_state = self._to_tensor(state_np)
            q_values_tensor = self.forward(t_state)
        
            # USE NUMPY
            q_values_np = q_values_tensor.detach().numpy()
            
            return np.argmax(q_values_np)

    def get_value(self, state_np, action):
        with torch.no_grad():
            t_state = self._to_tensor(state_np)
            q_values_tensor = self.forward(t_state)
            
            # convert to float
            return q_values_tensor[action].item()

    def update(self, state_np, action, target_val):
        t_state = self._to_tensor(state_np)
        t_target = torch.tensor(target_val, dtype=torch.float32)
        
        q_values = self.forward(t_state)
        prediction = q_values[action]

        loss = (prediction - t_target) ** 2
        
        self.net.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for param in self.net.parameters():
                if param.grad is not None:
                    param -= self.alpha * param.grad

class GridWorldQFunc:
    def __init__(self, alpha, input_dim=25, output_dim=4):
        self.weights = np.zeros((output_dim, input_dim))
        self.alpha = alpha

    def predict(self, state_np):
        # forward pass
        q_values = np.dot(self.weights, state_np)
        return np.argmax(q_values)

    def get_value(self, state_np, action):
        return np.dot(self.weights[action], state_np)

    def update(self, state_np, action, target_val):
        prediction = np.dot(self.weights[action], state_np)
        error = target_val - prediction
        self.weights[action] += self.alpha * error * state_np