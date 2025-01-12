import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from container import Container
from environment import BinPackingEnv

container = Container('Container', 10, 10)
max_items = 10

env = BinPackingEnv(container, max_items)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_binpacking")

model = PPO.load("ppo_binpacking")

obs = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    if info['success']:
        env.render()
    else:
        print(f"Failed: {info['reason']}")

    if done:
        print(f"Filling ratio: {env.container.get_filling_ratio()}")
        break

env.close()
