import gym
from stable_baselines3 import PPO

from item import Item
from container import Container
from environment import BinPackingEnv

container = Container('Container', 20, 20, 30)
items = []
items.append(Item("Item1", 10, 10, 10))
items.append(Item("Item2", 5, 5, 5))
items.append(Item("Item3", 5, 5, 5))
items.append(Item("Item4", 5, 5, 5))
items.append(Item("Item5", 5, 5, 5))


BinPackingEnv.register()
env = gym.make("BinPacking-v0", container=container, items=items)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save("ppo_binpacking")

model = PPO.load("ppo_binpacking")


obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    env.render()

    if done:
        break

env.close()
