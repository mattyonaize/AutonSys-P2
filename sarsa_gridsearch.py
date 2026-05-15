from env_setup import create_env
from sarsa_agent import SARSA

import numpy as np
import pandas as pd


alphas = [0.05, 0.1, 0.5]
gammas = [0.8, 0.9, 0.99]
epsilon_mins = [0.01, 0.05, 0.1]
epsilon_decays = [0.99, 0.995, 0.999, 0.9995, 0.9999]

runs_per_setting = 3
results = []
total_run_rewards = []

for alpha in alphas:
    for gamma in gammas:
        for epsilon_decay in epsilon_decays:
            for epsilon_min in epsilon_mins:
                for run in range(runs_per_setting):

                    print(f"alpha={alpha}")
                    print(f"gamma={gamma}")
                    print(f"epsilon_min={epsilon_min}")
                    print(f"epsilon_decay={epsilon_decay}")

                    env = create_env()

                    agent = SARSA(
                        state_size=env.observation_space.n,
                        action_size=env.action_space.n,
                        alpha=alpha,
                        gamma=gamma,
                        epsilon_min=epsilon_min,
                        epsilon_decay=epsilon_decay
                    )

                    rewards = agent.train(env, episodes=3000)
                    avg_reward = np.mean(rewards[-100:])

                    results.append({
                        "alpha": alpha,
                        "gamma": gamma,
                        "epsilon_min": epsilon_min,
                        "epsilon_decay": epsilon_decay,
                        "avg_reward": avg_reward
                    })

                    total_run_rewards.append(avg_reward)

                final_avg_reward = np.mean(total_run_rewards[-runs_per_setting:])

print("\nBeste hyperparameters:")

# Print de hyperparameters met de hoogste gemiddelde reward
best_result = max(results, key=lambda x: x['avg_reward'])
best_hyperparameters = {
    'alpha': best_result['alpha'],
    'gamma': best_result['gamma'],
    'epsilon_min': best_result['epsilon_min'],
    'epsilon_decay': best_result['epsilon_decay']
}

print(best_hyperparameters)

df = pd.DataFrame(results)
df.to_csv("sarsa_gridsearch_results.csv", index=False)