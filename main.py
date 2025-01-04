from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from item import generate_items
from container import Container
from environment import BinPackingEnv

container_size = (10, 10)
items_count = 5


def create_env(items_size: tuple[int, int, int]) -> BinPackingEnv:
    container = Container('Container', container_size[0], container_size[1])
    items = generate_items(items_size[0], items_size[1], items_size[2], items_count)
    return BinPackingEnv(container, items)


# env_fns = [
#     lambda: create_env((10, 10, 10))
# ]
# 
# envs = DummyVecEnv(env_fns)

envs = create_env((10, 10, 10))

model = PPO("MlpPolicy", envs, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_binpacking")

# model = PPO.load("ppo_binpacking")

test_env = create_env((10, 10, 10))
obs = test_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)

    if info['success']:
        test_env.render()

    if done:
        break

test_env.close()
