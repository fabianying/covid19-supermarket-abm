import math
import numpy as np

"""
Shelf class object
"""


class Shelf(object):

    def __init__(self, x, y, width, depth, name, angle, equipment):
        self.name = name

        self._width = width
        self._depth = depth
        self._x = x
        self._y = y

        self._angle = angle

        self.equipment = equipment
        self.units = 'cm'

    def convert_to_m(self):
        if self.units == 'cm':
            self._x /= 100
            self._y /= 100
            self._depth /= 100
            self._width /= 100
            self.units = 'm'

    def convert_to_cm(self):
        if self.units == 'm':
            self._x *= 100
            self._y *= 100
            self._depth *= 100
            self._width *= 100
            self.units = 'cm'

    @property
    def corners(self):
        return self.get_corners()

    @property
    def center(self):
        return self.get_centre()

    def get_corners(self):
        """
        get_corners returns an np array:
        np.c_[front_right_corner, back_right_corner, back_left_corner, front_left_corner]
        """
        angle = self._angle
        depth = self._depth
        width = self._width
        angle_in_radians = math.radians(angle)
        cos_angle = math.cos(angle_in_radians)
        sin_angle = math.sin(angle_in_radians)
        x1 = self._x - sin_angle * depth
        y1 = self._y + cos_angle * depth
        x2 = x1 + cos_angle * width
        y2 = y1 + sin_angle * width
        x3 = x2 + sin_angle * depth
        y3 = y2 - cos_angle * depth
        return np.array([[self._x, self._y], [x1, y1], [x2, y2], [x3, y3]])

    def get_centre(self):
        return self.corners.mean(axis=0)

    def __hash__(self):
        return hash(self.name)

    def _compare(self, other, method):
        if isinstance(other, Shelf):
            return method(self.name, other.name)
        else:
            return ValueError('Can only compare two shelves')

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)

    def __repr__(self):
        return 'Shelf({})'.format(self.name)
