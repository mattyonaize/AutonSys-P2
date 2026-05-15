import gymnasium as gym

def create_env(render_mode=None):
    env = gym.make("Taxi-v3", render_mode=render_mode)
    return env