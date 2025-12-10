import numpy as np

class GridWorldEnv:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.water_pos = (4, 2)
        
        self.walls = {(2, 2), (3, 2)} 

        self.current_pos = self.start_pos

        self.action_deltas = {
            0: (-1, 0),
            1: (1, 0), 
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self):
        self.current_pos = self.start_pos
        return self._get_state_index()

    def step(self, action):
        rand = np.random.random()
        move_action = None

        if rand < 0.80:
            move_action = action 
        elif rand < 0.85:
            move_action = self._veer(action, 'right')
        elif rand < 0.90:
            move_action = self._veer(action, 'left')
        else:
            move_action = None 

        if move_action is not None:
            delta = self.action_deltas[move_action]
            new_r = self.current_pos[0] + delta[0]
            new_c = self.current_pos[1] + delta[1]
            new_pos = (new_r, new_c)

            if (0 <= new_r < self.rows and 
                0 <= new_c < self.cols and 
                new_pos not in self.walls):
                self.current_pos = new_pos

        reward = 0.0
        done = False

        if self.current_pos == self.goal_pos:
            reward = 10.0
            done = True

        elif self.current_pos == self.water_pos:
            reward = -10.0
            done = False 
        
        return self._get_state_index(), reward, done

    def _get_state_index(self):

        return self.current_pos[0] * self.cols + self.current_pos[1]

    def _veer(self, action, direction):
        # 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0: 
            return 3 if direction == 'right' else 2
        elif action == 1:
            return 2 if direction == 'right' else 3
        elif action == 2: 
            return 0 if direction == 'right' else 1
        elif action == 3:
            return 1 if direction == 'right' else 0
        return action

def one_hot(state_index, num_states=25):
    vec = np.zeros(num_states, dtype=np.float32)
    vec[state_index] = 1.0
    return vec