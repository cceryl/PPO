from item import Item

import matplotlib.pyplot as plt
import numpy as np


class Container:
    def __init__(self, name: str, length: int, width: int):
        self.name: str = name
        self.length: int = length
        self.width: int = width
        self.height: int = 0
        self.height_map: np.ndarray = np.zeros((length, width), dtype=int)
        self.items: list[Item] = []

    def reset(self):
        self.height = 0
        self.height_map = np.zeros((self.length, self.width), dtype=int)
        self.items = []

    def get_volume(self) -> int:
        return self.length * self.width * self.height

    def get_filling_ratio(self) -> float:
        return sum([item.get_volume() for item in self.items]) / self.get_volume()

    def check_item_fit(self, item: Item, position: list[int]) -> bool:
        """
        Check if the item fits in the container at the given position.
        """

        length, width, height = item.get_dimension()
        x, y, z = position

        if x < 0 or y < 0 or z < 0:
            return False

        if x + length > self.length or y + width > self.width:
            return False

        return True

    def add_item(self, item: Item, position: list[int]) -> bool:
        if not self.check_item_fit(item, position):
            return False

        length, width, height = item.get_dimension()
        x, y, z = position

        z = np.max([self.height_map[x:x + length, y:y + width]])
        item.position = [x, y, z]

        self.items.append(item)

        self.height_map[x:x+length, y:y+width] = z + height
        self.height = max(self.height, z + height)

        return True

    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for item in self.items:
            x, y, z = item.position
            length, width, height = item.get_dimension()
            ax.bar3d(x, y, z, length, width, height, edgecolor='black')

        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)
        ax.set_zlim(0, self.height)

    def string(self) -> str:
        return "%s(%sx%sx%s) vol(%s) items(%s) filling_ratio(%s)" % (
            self.name, self.length, self.width, self.height,
            self.get_volume(), len(self.items), self.get_filling_ratio()
        )
