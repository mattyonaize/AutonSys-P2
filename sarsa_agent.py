import numpy as np

class SARSA:

    def __init__(
            self,
            state_size,
            action_size,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.1,
            epsilon_decay=0.99,
            epsilon_min=0.01
            ):
        
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size))

    # Exploration-exploitation hyperparameter
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]

        # SARSA update
        self.q_table[state, action] += current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=1000, max_steps=100):

        rewards_per_episode = []

        for episode in range(episodes):
            state, info = env.reset()
            action = self.choose_action(state)
            total_reward = 0

            for step in range(max_steps):
                next_state, reward, terminated, truncated, info = env.step(action)
                next_action = self.choose_action(next_state)

                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

                total_reward += reward

                if terminated or truncated:
                    break

            self.decay_epsilon()

            rewards_per_episode.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])

                print(
                    f"Episode {episode + 1}"
                    f" - Gemiddelde Reward: {avg_reward:.2f}"
                    f" - Epsilon: {self.epsilon:.4f}"
                )

        return rewards_per_episode
    
    def test(self, env, episodes=10):
        for episode in range(episodes):
            state, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                total_reward += reward
                done = terminated or truncated
            print(f"Test Episode {episode + 1}: Reward: {total_reward}")