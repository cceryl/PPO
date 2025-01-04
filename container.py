from item import Item

import matplotlib.pyplot as plt
import numpy as np


class Container:
    def __init__(self, name: str, length: int, width: int):
        self.name: str = name
        self.length: int = length
        self.width: int = width
        self.height: int = 0
        self.height_map = np.zeros((length, width), dtype=int)
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

        length, width, _ = item.get_dimension()
        x, y, _ = position

        if x < 0 or x + length > self.length or y < 0 or y + width > self.width:
            return False

        z = np.max(self.height_map[x:x + length, y:y + width])
        item.position = [x, y, z]

        return all([not item.overlap(other_item) for other_item in self.items])

    def add_item(self, item: Item, position: list[int]) -> bool:
        if not self.check_item_fit(item, position):
            return False

        self.items.append(item)

        length, width, height = item.get_dimension()
        x, y, z = item.position

        self.height_map[x:x + length, y:y + width] = z + height
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

        ax.text2D(0.05, 0.95, "Filling ratio: %.2f" % self.get_filling_ratio(), transform=ax.transAxes)

    def string(self) -> str:
        return "%s(%sx%sx%s) vol(%s) items(%s) filling_ratio(%s)" % (
            self.name, self.length, self.width, self.height,
            self.get_volume(), len(self.items), self.get_filling_ratio()
        )
