import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from item import generate_items
from container import Container
from environment import BinPackingEnv

container = Container('Container', 10, 10)
items = generate_items(8, 8, 8, 8)

container_size = (10, 10)
items_count = 8


def create_env(items_size):
    container = Container('Container', *container_size)
    items = generate_items(*items_size, items_count)
    return BinPackingEnv(container, items)


envs = [
    lambda: create_env((8, 8, 8)),
    lambda: create_env((8, 8, 8)),
    lambda: create_env((8, 8, 8)),
    lambda: create_env((8, 8, 8)),
    lambda: create_env((8, 8, 8))
]

env = DummyVecEnv(envs)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_binpacking")

model = PPO.load("ppo_binpacking")

test_env = create_env((8, 8, 8))

obs = test_env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)

    if info['success']:
        test_env.render()

    if done:
        print(f"Filling ratio: {test_env.container.get_filling_ratio()}")
        break

test_env.close()
