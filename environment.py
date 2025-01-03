from constants import RotationType
from item import Item
from container import Container
from encoder import Encoder
from decoder import Decoder

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class BinPackingEnv(gym.Env):
    def __init__(self, container: Container, items: list[Item]):
        super(BinPackingEnv, self).__init__()

        self.state = None
        self.action_space = spaces.Discrete(len(items) * container.get_volume() * len(RotationType.All))
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            container.length * container.width * 3 + len(items) * 3,))
        self.reward_range = (0, 1)
        self.done = False

        self.container = container
        self.items = items.copy()
        self.inserted_items = []
        self.available_items = items.copy()
        self.last_filling_ratio = 0

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.state = self.encoder.encode_state(self.container, self.available_items,
                                               len(self.items) - len(self.available_items))

    def reset(self):
        self.container.reset()
        self.inserted_items = []
        self.available_items = self.items.copy()
        self.last_filling_ratio = 0

        self.state = self.encoder.encode_state(self.container, self.available_items,
                                               len(self.items) - len(self.available_items))

        self.done = False

        return self.state

    def step(self, action):
        decoder = Decoder()
        index, rotation, position = decoder.decode_action(action, self.container)

        if index >= len(self.available_items):
            return self.state, -1, self.done, {}

        item = self.available_items[index]
        item.rotation_type = rotation

        success = self.container.add_item(item, position)
        if not success:
            return self.state, -1, self.done, {}

        self.inserted_items.append(item)
        self.available_items.remove(item)

        if len(self.available_items) == 0:
            self.done = True

        filling_ratio_now = self.container.get_filling_ratio()
        reward = filling_ratio_now - self.last_filling_ratio
        self.last_filling_ratio = filling_ratio_now

        self.state = self.encoder.encode_state(self.container, self.available_items,
                                               len(self.items) - len(self.available_items))

        return self.state, reward, self.done, {}

    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for item in self.inserted_items:
            x, y, z = item.position
            length, width, height = item.get_dimension()
            ax.bar3d(x, y, z, length, width, height)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([0, self.container.length])
        ax.set_ylim([0, self.container.width])
        ax.set_zlim([0, self.container.height])

        plt.show()

    def register():
        gym.envs.register(
            id='BinPacking-v0',
            entry_point='environment:BinPackingEnv',
        )
