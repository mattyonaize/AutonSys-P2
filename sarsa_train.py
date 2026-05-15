from env_setup import create_env
from sarsa_agent import SARSA

import matplotlib.pyplot as plt

env = create_env()

state_size = env.observation_space.n
action_size = env.action_space.n

agent = SARSA(state_size=state_size, action_size=action_size)

rewards = agent.train(env, episodes=10000, max_steps=200)
test_env = create_env(render_mode='human')
agent.test(test_env)

# Visualiseer rewards over episodes
plt.plot(rewards)

plt.xlabel('Episode')
plt.ylabel('Gemiddelde Reward')
plt.title('SARSA Leercurve')
plt.savefig('SARSA_leercurve.png')