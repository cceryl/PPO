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
        self.action_space = spaces.MultiDiscrete([len(items), 6, container.length, container.width])
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            container.length * container.width * 3 + len(items) * 3,), dtype=np.int32)
        self.reward_range = (-1, 1)
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
        index, rotation, position = decoder.decode_action(action)

        if index >= len(self.available_items):
            return self.state, -1, self.done, {'success': False}

        item = self.available_items[index]
        item.rotation_type = rotation

        success = self.container.add_item(item, position)
        if not success:
            return self.state, -1, self.done, {'success': False}

        self.inserted_items.append(item)
        self.available_items.remove(item)

        if len(self.available_items) == 0:
            self.done = True

        filling_ratio = self.container.get_filling_ratio()
        reward = filling_ratio - self.last_filling_ratio
        self.last_filling_ratio = filling_ratio

        self.state = self.encoder.encode_state(self.container, self.available_items,
                                               len(self.items) - len(self.available_items))

        return self.state, reward, self.done, {'success': True}

    def render(self):
        self.container.render()
        plt.show()
