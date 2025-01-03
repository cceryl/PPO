from constants import RotationType


class Item:
    def __init__(self, name: str, length: int, width: int, height: int):
        self.name: str = name
        self.length: int = length
        self.width: int = width
        self.height: int = height
        self.rotation_type: RotationType = RotationType.Rotate_LWH
        self.position: list[int] = [0, 0, 0]

    def get_volume(self) -> int:
        return self.length * self.width * self.height

    def get_dimension(self) -> tuple:
        """
        Get the real dimension of the item according to the rotation type.
        """

        return {
            RotationType.Rotate_LWH: (self.length, self.width, self.height),
            RotationType.Rotate_LHW: (self.length, self.height, self.width),
            RotationType.Rotate_WHL: (self.width, self.height, self.length),
            RotationType.Rotate_WLH: (self.width, self.length, self.height),
            RotationType.Rotate_HLW: (self.height, self.length, self.width),
            RotationType.Rotate_HWL: (self.height, self.width, self.length)
        }[self.rotation_type]

    def overlap(self, other_item) -> bool:
        """
        Check if the item overlaps with another item.
        """

        length, width, height = self.get_dimension()
        x, y, z = self.position
        other_length, other_width, other_height = other_item.get_dimension()
        other_x, other_y, other_z = other_item.position

        if x + length <= other_x or other_x + other_length <= x:
            return False

        if y + width <= other_y or other_y + other_width <= y:
            return False

        if z + height <= other_z or other_z + other_height <= z:
            return False

        return True

    def string(self) -> str:
        return "(%sx%sx%s) pos(%s) rot(%s) vol(%s)" % (
            self.length, self.width, self.height,
            self.position, self.rotation_type, self.get_volume()
        )
