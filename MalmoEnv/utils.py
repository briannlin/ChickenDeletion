import math
from typing import Tuple

def decimal_round(x, decimal_precision: float, round_precision=1) -> float:
    return round(round(x * (1 / decimal_precision)) / (1 / decimal_precision), round_precision)



def integer_round(x, integer_precision: int) -> int:
    return round(x / integer_precision) * integer_precision


def distance(xy: Tuple[float, float]) -> float:
    return round((xy[0] ** 2 + xy[1] ** 2) ** (1 / 2), 2)


def calculate_target_angle(xy: Tuple[float, float]):
    angle = math.atan2(xy[0], xy[1])
    angle_in_degrees = math.degrees(angle)
    return (360 - angle_in_degrees) % 360


def angle_of_depression(height, distance):
    angle_radians = math.atan2(distance, height)
    angle_degrees = math.degrees(angle_radians)

    return 90 - angle_degrees
