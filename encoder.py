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
        Encode the container by (height, higher_len, higher_wid, pos_len, neg_len, pos_wid, neg_wid) for each cell on a 2D plane.
        'height' is the height of the highest item that covers the cell.
        'higher_len' is the distance to the nearest higher cell in the length direction.
        'higher_wid' is the distance to the nearest higher cell in the width direction.
        'pos_len' is the distance to the edge of the plane in the positive length direction.
        'neg_len' is the distance to the edge of the plane in the negative length direction.
        'pos_wid' is the distance to the edge of the plane in the positive width direction.
        'neg_wid' is the distance to the edge of the plane in the negative width direction.
        """

        encoded = np.zeros((container.length, container.width, 7), dtype=int)
        for x in range(container.length):
            for y in range(container.width):
                h = container.height_map[x, y]

                higher_len = 0
                for i in range(x, container.length):
                    if container.height_map[i, y] > h:
                        break
                    higher_len += 1

                higher_wid = 0
                for j in range(y, container.width):
                    if container.height_map[x, j] > h:
                        break
                    higher_wid += 1

                pos_len = 0
                for i in range(x, container.length):
                    if container.height_map[i, y] != h:
                        break
                    pos_len += 1

                neg_len = 0
                for i in range(x, -1, -1):
                    if container.height_map[i, y] != h:
                        break
                    neg_len += 1

                pos_wid = 0
                for j in range(y, container.width):
                    if container.height_map[x, j] != h:
                        break
                    pos_wid += 1

                neg_wid = 0
                for j in range(y, -1, -1):
                    if container.height_map[x, j] != h:
                        break
                    neg_wid += 1

                encoded[x, y] = np.array([h, higher_len, higher_wid, pos_len, neg_len, pos_wid, neg_wid])

        return encoded.flatten().astype(int)

    def encode_state(self, container: Container, items: list[Item], padding: int = 0) -> np.ndarray:
        items = items + [Item('', 0, 0, 0)] * padding
        return np.concatenate([self.encode_items(items), self.encode_container(container)]).astype(int)
