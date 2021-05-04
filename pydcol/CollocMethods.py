"""
Enums representing set of supported collocation methods

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

TRAP = 0
HERM = 1
RADAU = 2
EB = 3
EF = 4

MIDPOINT_METHODS = [HERM, RADAU]

METHOD_NAMES = ["Trapezoid", "Hermite-Simpson", "Radau", "Euler-Backward", "Euler-Forward"]
