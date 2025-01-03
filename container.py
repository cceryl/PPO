from item import Item


class Container:
    def __init__(self, name: str, length: int, width: int, height: int):
        self.name: str = name
        self.length: int = length
        self.width: int = width
        self.height: int = height
        self.items: list[Item] = []

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

        if x + length > self.length or y + width > self.width or z + height > self.height:
            return False

        return all([not item.overlap(other_item) for other_item in self.items])

    def add_item(self, item: Item, position: list[int]) -> bool:
        if self.check_item_fit(item, position):
            item.position = position
            self.items.append(item)
            return True
        return False

    def string(self) -> str:
        return "%s(%sx%sx%s) vol(%s) items(%s) filling_ratio(%s)" % (
            self.name, self.length, self.width, self.height,
            self.get_volume(), len(self.items), self.get_filling_ratio()
        )
