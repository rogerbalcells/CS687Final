import numpy as np

class MountainCarEnv:
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.5
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        self.goal_position = 0.5
        
        self.force = 0.001
        self.gravity = 0.0025
        
        self.max_steps = 1000  
        self.done_steps = 0

        self.state = None
        self.reset()

    def reset(self):
        self.done_steps = 0
        
        start_pos = np.random.uniform(-0.6, -0.4)
        start_vel = 0.0
        self.state = np.array([start_pos, start_vel], dtype=np.float32)
        return self.state

    def step(self, action_idx):
        position, velocity = self.state
        self.done_steps += 1
        
        # 0 -> -1 (Reverse)
        # 1 ->  0 (Neutral)
        # 2 ->  1 (Forward)
        action_val = action_idx - 1 
        
        velocity += (self.force * action_val) - (self.gravity * np.cos(3 * position))
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)
        
        position += velocity
        
        if position <= self.min_position:
            position = self.min_position
            velocity = 0.0
        elif position >= self.max_position:
            position = self.max_position
            velocity = 0.0
            
        self.state = np.array([position, velocity], dtype=np.float32)
        
        reached_goal = (position >= self.goal_position)
        timeout = (self.done_steps >= self.max_steps)
        
        done = reached_goal or timeout
        
        reward = 0.0 if reached_goal else -1.0
        
        return self.state, reward, done