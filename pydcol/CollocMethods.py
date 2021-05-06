"""
Enums representing set of supported collocation methods

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

TRAP = 0 # Trapezoid method, 1st order accurate
HERM = 1 # Hermite-Simpson method, 4th order accurate
RADAU = 2 # Radau IIA method, 3rd order accurate
EB = 3 # Euler Backward method, 1st order accurate
EF = 4 # Euler Forward method, 1st order accurate

# midpoint methods are treated differently
MIDPOINT_METHODS = [HERM, RADAU]

# Useful to be able to map the methods to their names as strings
METHOD_NAMES = ["Trapezoid", "Hermite-Simpson", "Radau", "Euler-Backward", "Euler-Forward"]
