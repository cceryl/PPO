from item import Item
from container import Container

import numpy as np


class Encoder:
    def __init__(self):
        pass

    def encode_item(self, item: Item) -> np.ndarray:
        return np.array([item.length, item.width, item.height]).flatten().astype(int)

    def encode_items(self, items: list[Item]) -> np.ndarray:
        return np.array([self.encode_item(item) for item in items]).flatten().astype(int)

    def encode_container(self, container: Container) -> np.ndarray:
        """
        Encode the container by (height, dist_near_len, dist_near_wid) for each cell on a 2D plane.
        'height' is the height of the highest item that covers the cell.
        'dist_near_len' is the distance to the nearest higher cell in the length direction.
        'dist_near_wid' is the distance to the nearest higher cell in the width direction.
        """

        encoded = np.zeros((container.length, container.width, 3), dtype=int)
        for x in range(container.length):
            for y in range(container.width):
                h = container.height_map[x, y]

                dist_near_len = np.count_nonzero(container.height_map[x:, y] <= h)
                dist_near_wid = np.count_nonzero(container.height_map[x, y:] <= h)

                encoded[x, y] = [h, dist_near_len, dist_near_wid]

        return encoded.flatten().astype(int)

    def encode_state(self, container: Container, items: list[Item], padding: int = 0) -> np.ndarray:
        items = items + [Item('', 0, 0, 0)] * padding
        return np.concatenate([self.encode_items(items), self.encode_container(container)]).astype(int)  
