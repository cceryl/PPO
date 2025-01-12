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
        self.observation_space = spaces.Dict({
            'container': spaces.Box(
                low=np.zeros((container.length, container.width, 7), dtype=np.float32),
                high=np.full((container.length, container.width, 7), [
                    container.height, container.length, container.width, container.length, container.length, container.width, container.width
                ], dtype=np.float32),
                dtype=np.float32
            ),
            'items_obs': spaces.Box(low=0, high=max(container.length, container.width, container.height), shape=(len(items), 3), dtype=np.float32),
            'item_mask': spaces.Box(low=0, high=1, shape=(len(items),), dtype=bool)
        })
        self.reward_range = (0, 1)
        self.done = False

        self.container = container
        self.items = items.copy()
        self.inserted_items = []
        self.available_items = items.copy()
        self.last_filling_ratio = 0
        self.last_height = 0
        self.last_average_height = 0

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.state = self.encoder.encode_state(self.container, self.available_items, len(self.items))

    def reset(self):
        self.container.reset()
        self.inserted_items = []
        self.available_items = self.items.copy()
        self.last_filling_ratio = 0
        self.last_height = 0
        self.last_average_height = 0

        self.state = self.encoder.encode_state(self.container, self.available_items, len(self.items))

        self.done = False

        return self.state

    def step(self, action: list[int]):
        decoder = Decoder()
        index, rotation, position = decoder.decode_action(action)

        if index >= len(self.available_items):
            return self.state, 0, self.done, {'success': False}

        item = self.available_items[index]
        item.rotation_type = rotation

        success = self.container.add_item(item, position)
        if not success:
            return self.state, 0, self.done, {'success': False}

        self.inserted_items.append(item)
        self.available_items.remove(item)

        if len(self.available_items) == 0:
            self.done = True

        filling_ratio = self.container.get_filling_ratio()
        height = self.container.height
        average_height = sum([(item.position[2] + item.get_dimension()[2])
                             for item in self.inserted_items]) / len(self.inserted_items)

        reward = (filling_ratio - self.last_filling_ratio) * 10 - \
            (height - self.last_height) * 0.1 - \
            (average_height - self.last_average_height) * 0.1

        self.last_filling_ratio = filling_ratio
        self.last_height = self.container.height
        self.last_average_height = average_height

        self.state = self.encoder.encode_state(self.container, self.available_items, len(self.items))

        return self.state, reward, self.done, {'success': True}

    def render(self):
        self.container.render()
        plt.show()
