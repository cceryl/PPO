from item import Item
from container import Container

import numpy as np


class Encoder:
    def __init__(self):
        pass

    def encode_item(self, item: Item) -> np.ndarray:
        return np.array([item.length, item.width, item.height])

    def encode_items(self, items: list[Item]) -> np.ndarray:
        return np.array([self.encode_item(item) for item in items])

    def encode_container(self, container: Container) -> np.ndarray:
        """
        Encode the container by (height, dist_near_len, dist_near_wid) for each cell on a 2D plane.
        'height' is the height of the highest item that covers the cell.
        'dist_near_len' is the distance to the nearest higher cell in the length direction.
        'dist_near_wid' is the distance to the nearest higher cell in the width direction.
        """

        height_map = np.zeros((container.length, container.width), dtype=int)
        for item in container.items:
            x, y, z = item.position
            length, width, height = item.get_dimension()
            height_map[x:x+length, y:y+width] = max(height_map[x:x+length, y:y+width], z + height)

        encoded = np.zeros((container.length, container.width, 3), dtype=int)
        for x in range(container.length):
            for y in range(container.width):
                h = height_map[x, y]
                if h == 0:
                    continue

                dist_near_len = 0
                for i in range(x, container.length):
                    if height_map[i, y] > h:
                        break
                    dist_near_len += 1

                dist_near_wid = 0
                for j in range(y, container.width):
                    if height_map[x, j] > h:
                        break
                    dist_near_wid += 1

                encoded[x, y] = np.array([h, dist_near_len, dist_near_wid])

        return encoded
