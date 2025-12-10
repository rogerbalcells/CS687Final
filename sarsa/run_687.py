import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorldEnv, one_hot
from agent import NStepSARSAAgent
from policy import GridWorldQFunc

RUNS = 5       
EP_CNT = 500 

params = [
    {"n": 13, "alpha": 0.07, "epsilon": 0.24},    
    {"n": 14, "alpha": 0.03, "epsilon": 0.28},
    {"n": 15, "alpha": 0.04, "epsilon": 0.22},
    {"n": 15, "alpha": 0.03, "epsilon": 0.28},
    {"n": 15, "alpha": 0.03, "epsilon": 0.24},
]

def run_gridworld(n_step, alpha, epsilon, run_cnt, num_episodes):
    returns = np.zeros((run_cnt, num_episodes))
    
    gamma = 0.9

    for run in range(run_cnt):
        env = GridWorldEnv()
        
        model = GridWorldQFunc(input_dim=25, output_dim=4, alpha=alpha)
        
        agent = NStepSARSAAgent(model, action_cnt=4, 
                                gamma=gamma, n_step=n_step, epsilon=epsilon)
        
        for ep in range(num_episodes):
            state_idx = env.reset()
            state_vec = one_hot(state_idx)
            
            action = agent.select_action(state_vec)
            agent.save_transition(state_vec, action, 0)
            
            t = 0
            T = float('inf')
            ep_return = 0
            discount = 1.0        
            
            while True:
                if t < T:
                    next_idx, reward, done = env.step(action)
                    ep_return += discount * reward
                    discount *= gamma

                    if t >= 1000 and not done:
                        done = True
                    
                    if done:
                        T = t + 1

                        ps, pa, _ = agent.buffer[-1]
                        agent.buffer[-1] = (ps, pa, reward)
                        
                        next_vec = one_hot(next_idx)
                        agent.save_transition(next_vec, 0, 0)
                    else:
                        next_vec = one_hot(next_idx)
                        next_action = agent.select_action(next_vec)

                        ps, pa, _ = agent.buffer[-1]
                        agent.buffer[-1] = (ps, pa, reward)
                        
                        agent.save_transition(next_vec, next_action, 0)
                        action = next_action
                
                tau = t - n_step + 1
                if tau >= 0:
                    agent.update(t, T)
                
                if tau == T - 1:
                    break
                t += 1
            
            returns[run, ep] = ep_return
            agent.clear_buffer()

            if (ep+1) % 100 == 0:
                print(f"run {run+1}/{run_cnt} - Ep {ep+1} - return: {ep_return:.2f}")

    return returns

if __name__ == "__main__":
    
    for p in params:
        print(f"-------- RUNNING: {p}")

        rets = run_gridworld(
            n_step=p['n'], 
            alpha=p['alpha'], 
            epsilon=p['epsilon'], 
            run_cnt=RUNS, 
            num_episodes=EP_CNT
        )
        
        mean_curve = np.mean(rets, axis=0)
        std_curve = np.std(rets, axis=0)
        
        plt.figure(figsize=(10,6))
        x_axis = np.arange(EP_CNT)
        
        plt.plot(x_axis, mean_curve, label=f"Mean Return")
        plt.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        
        final_avg = np.mean(mean_curve)
        title_str = (f"GridWorld: n={p['n']}, alpha={p['alpha']}, eps={p['epsilon']}\n"
                     f"Final 50 Ep Avg: {final_avg:.2f}")
        
        plt.title(title_str)
        plt.xlabel("Episode")
        plt.ylabel("Discounted Return")
        plt.grid(True, alpha=0.3)
        
        filename = f"gridworld_n{p['n']}_a{p['alpha']}_e{p['epsilon']}.png"
        plt.savefig(filename)
        plt.close()