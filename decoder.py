from constants import RotationType
from container import Container

import numpy as np


class Decoder:
    def __init__(self):
        pass

    def decode_action(self, action: int, container: Container) -> tuple[int, RotationType, tuple[int, int, int]]:
        """
        Decode the action space to (index, rotation, position).
        """

        index = action // (len(RotationType.All) * container.get_volume())
        action %= len(RotationType.All) * container.get_volume()

        rotation = RotationType.All[action // container.get_volume()]
        position_code = action % container.get_volume()

        position_x = position_code // (container.width * container.height)
        position_code %= container.width * container.height
        position_y = position_code // container.height
        position_z = position_code % container.height

        return index, rotation, (position_x, position_y, position_z)
