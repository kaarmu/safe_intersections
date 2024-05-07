from dataclasses import dataclass
from math import hypot, cos, sin

import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    @classmethod
    def origin(cls):
        return cls(0, 0)
    
    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1])

    def normalize(self):
        return self / abs(self)

    def to_array(self):
        return np.array([self.x, self.y])

    def __abs__(self):
        return hypot(self.x, self.y)

    def __neg__(self):
        _type = type(self)
        return _type(-self.x, -self.y)

    def __add__(self, other):
        _type = type(self)
        if type(other) == _type:
            return _type(self.x + other.x, self.y + other.y)
        else:
            return NotImplemented
        
    def __sub__(self, other):
        _type = type(self)
        if type(other) == _type:
            return _type(self.x - other.x, self.y - other.y)
        else:
            return NotImplemented

@dataclass(frozen=True, slots=True)
class CurveSegment:
    length: float
    curvature: float

    @property
    def end(self) -> Point:
        return Point(self.length * cos(self.curvature),
                     self.length * sin(self.curvature))
    
    def __len__(self):
        return self.length
    
    def to_array(self, density=50, start=Point.origin()):
        n = int(self.length * density) + 1
        out = np.zeros((n, 2))

        if self.curvature == 0:
            out[:, 0] = np.linspace(0, self.length, n)
        else:
            radius = 1 / self.curvature
            ang_step = self.length / radius / (n - 1)
            ix = np.arange(n)
            out[:, 0] = np.sin(ix * ang_step)
            out[:, 1] = 1 - np.cos(ix * ang_step)
        
        out += start.to_array()
        return out
    
class Line(CurveSegment):
    def __init__(self, length):
        super().__init__(length, 0)

class Arc(CurveSegment):
    def __init__(self, radius, angle):
        super().__init__(radius*angle, 1/radius)

@dataclass(frozen=True, slots=True)
class Curve:
    start: Point
    angle: float
    segments: tuple = ()

    @property
    def end(self):
        end = sum((segm.end for segm in self.segments), start=Point.origin())
        end -= self.start
        end = Point.from_array(self._rotm @ end.to_array())
        end += self.start
        return end
    
    @property
    def _rotm(self):
        return np.array([[+np.cos(self.angle), +np.sin(self.angle)], 
                         [-np.sin(self.angle), +np.cos(self.angle)]])
    
    @property
    def junctions(self):
        junct = self.start
        yield junct
        for segm in self.segments:
            junct += segm.end
            yield junct

    def to_array(self, density=50):
        out = np.concatenate([start.to_array() + segm.to_array(density) 
                              for start, segm in zip(self.junctions, self.segments)])
        out = np.dot(out, self._rotm)
        return out

    def __len__(self):
        return sum(map(len, self.segments))
    
    def __add__(self, other):
        if isinstance(other, CurveSegment):
            return Curve(self.start, self.angle, self.segments + (other,))
        else:
            return NotImplemented