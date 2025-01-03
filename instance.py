# This file generates the problem instances for BPP problems.

import random


def generate_instance(length: int, width: int, height: int, num_items: int) -> list[list[int, int, int]]:
    """
    Generate a problem instance for the BPP problem.
    :param length: The length of the bin.
    :param width: The width of the bin.
    :param height: The height of the bin.
    :param num_items: The number of items.
    :return: A list of items, each item is a list of three integers [length, width, height].
    """

    items = [[length, width, height]]

    while len(items) < num_items:
        # The probability of choosing an item is proportional to the volume of the item
        item = items.pop(random.choices(
            range(len(items)),
            weights=[item[0] * item[1] * item[2] for item in items],
            k=1
        )[0])

        # The probability of choosing an axis is proportional to the length of the axis
        axis = random.choices(
            [0, 1, 2],
            weights=[item[0], item[1], item[2]],
            k=1
        )[0]

        # The closer to the center, the higher the probability
        position = random.normalvariate(0.5, 0.1)
        position = int(position * item[axis])
        position = max(1, min(position, item[axis] - 1))

        item1 = item.copy()
        item2 = item.copy()
        item1[axis] = position
        item2[axis] = item[axis] - position

        item1[0], item1[1], item1[2] = random.sample([item1[0], item1[1], item1[2]], 3)
        item2[0], item2[1], item2[2] = random.sample([item2[0], item2[1], item2[2]], 3)

        items.append(item1)
        items.append(item2)

    return items
