import math

from bloqade.gemini.star import (
    DEFAULT_STEANE_STAR_SUPPORT as DEFAULT_STEANE_STAR_SUPPORT,
    VALID_STEANE_STAR_SUPPORTS as VALID_STEANE_STAR_SUPPORTS,
    validate_steane_star_support as validate_steane_star_support,
)


def steane_star_theta(theta: float) -> float:
    magnitude = 2 * math.atan(abs(math.tan(theta / 2)) ** (1 / 3))
    return -math.copysign(magnitude, theta)
