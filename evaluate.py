from container import Container
from environment import BinPackingTestEnv
from item import generate_items

import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


container_size = (10, 10)
items = generate_items(8, 8, 8, 10)
tree_search = False
tree_split_factor = 2
tree_split_depth = 3

env = BinPackingTestEnv(Container('Container', container_size[0], container_size[1]), items)

best_model_steps = 0
for file in os.listdir('./models/'):
    if file.startswith('ppo_binpacking') and file.endswith('.zip'):
        steps = int(file.split('_')[2])
        if steps > best_model_steps:
            best_model_steps = steps

best_model_path = f'./models/ppo_binpacking_{best_model_steps}_steps.zip'
model = PPO.load(best_model_path)

if tree_search:
    env.reset()
    test_envs = [env]
    current_depth = 0

    while current_depth < tree_split_depth:
        new_envs = []
        for env in test_envs:
            for _ in range(tree_split_factor):
                copy = env.copy()
                obs = copy.state
                action, _ = model.predict(obs)
                copy.step(action)
                new_envs.append(copy)
        
        test_envs = new_envs
        current_depth += 1

    results: list[tuple[BinPackingTestEnv, float]] = []

    for env in test_envs:
        obs = env.state
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)

        results.append((env, env.container.get_filling_ratio()))

    results.sort(key=lambda x: x[1], reverse=True)
    for i, (env, ratio) in enumerate(results):
        print(f"Result {i + 1}: Filling ratio: {ratio}")
        env.container.render()

else:
    obs = env.reset()
    done = False

    print(f"items: {len(env.items)}")

    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if info['success']:
            env.render()
        else:
            print(f"failure: {info['reason']}")

    print(f"filling ratio: {env.container.get_filling_ratio()}")
