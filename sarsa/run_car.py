import numpy as np
import matplotlib.pyplot as plt

from mountain_car import MountainCarEnv
from agent import NStepSARSAAgent
from policy import MountainCarNet 

RUNS = 5
EP_CNT = 100 

params = [    
    {"n": 20, "alpha": 0.000015, "epsilon": 0.02, "layers": [4, 4]},
    {"n": 20, "alpha": 0.000015, "epsilon": 0.02, "layers": [2, 2]},

    {"n": 20, "alpha": 0.00002, "epsilon": 0.02, "layers": [4, 4]},
    {"n": 20, "alpha": 0.00005, "epsilon": 0.02, "layers": [4, 4]},

    {"n": 18, "alpha": 0.000015, "epsilon": 0.02, "layers": [4, 4]},
]

results = {}

def run_mountain_car(step, alpha, epsilon, layers):

    returns = np.zeros((RUNS, EP_CNT))    
    gamma = 1.0

    for run in range(RUNS):
        env = MountainCarEnv()
        
        model = MountainCarNet(layers=layers, alpha=alpha)
        
        agent = NStepSARSAAgent(model, action_cnt=3, 
                                gamma=gamma, n_step=step, epsilon=epsilon)
        
        for ep in range(EP_CNT):
            state = env.reset()
            
            action = agent.select_action(state)
            agent.save_transition(state, action, 0)
            
            t = 0
            T = float('inf')
            ep_return = 0            

            while True:
                if t < T:
                    next_state, reward, done = env.step(action)
                    ep_return += reward 
                    
                    if done:
                        T = t + 1

                        ps, pa, _ = agent.buffer[-1]
                        agent.buffer[-1] = (ps, pa, reward)
                        
                        agent.save_transition(next_state, 0, 0)
                    else:
                        next_action = agent.select_action(next_state)
                        
                        ps, pa, _ = agent.buffer[-1]
                        agent.buffer[-1] = (ps, pa, reward)
                        
                        agent.save_transition(next_state, next_action, 0)
                        action = next_action
                        
                tau = t - step + 1
                if tau >= 0:
                    agent.update(t, T)
                
                if tau == T - 1:
                    break
                t += 1
            
            returns[run, ep] = ep_return
            agent.clear_buffer()
            
            if (ep+1) % 20 == 0:
                print(f"run {run+1}, Ep {ep+1}, return: {ep_return}")

    return returns

if __name__ == "__main__":

    for p in params:
        print(f"-------- Running: {p}")
        
        all_returns = run_mountain_car(p['n'], p['alpha'], p['epsilon'], p['layers'])
        
        mean_ret = np.mean(all_returns, axis=0)
        std_ret = np.std(all_returns, axis=0)
        x = np.arange(len(mean_ret))

        overall_avg = np.mean(all_returns)        
        
        plt.figure(figsize=(10,6))
        plt.plot(x, mean_ret, label=f'Mean Return (Avg Return: {overall_avg:.2f})')
        plt.fill_between(x, mean_ret - std_ret, mean_ret + std_ret, color='blue', alpha=0.3)
        
        title_str = (f"Params: n={p['n']}, alpha={p['alpha']}, epsilon={p['epsilon']}, layers={p['layers']}\n"
                     f"Average Return: {overall_avg:.2f} (Last 50 epsiodes average: {np.mean(all_returns[:, -50:])})")
        
        plt.title(title_str)
        plt.xlabel("Episode")
        plt.ylabel("Return")

        layers_str = str(p['layers']).replace('[','').replace(']','').replace(', ','-')
        filename = f"mountain_car_n{p['n']}_alpha{p['alpha']}_epsilon{p['epsilon']}_layers{layers_str}.png"
        plt.savefig(filename)
        plt.close()