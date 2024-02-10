import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

model_dir = "models/PPO"


env = gym.make("LunarLander-v2" , render_mode="rgb_array")

env.reset()

model_path = f"{model_dir}/290000.zip"

model = PPO.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        env.render() 
        action, _ = model.predict(obs) 
        obs, reward, done, info, _ = env.step(action)
        # print(reward)

env.close()