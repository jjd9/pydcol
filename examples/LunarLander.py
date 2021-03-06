"""

Lunar lander example

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/02/2021
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix

from pydcol.Animator import draw_lander
from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

if __name__ == "__main__":

	colloc_method = HERM

	# define variables
	x, xdot, y, ydot, th, thdot = symbols("x xdot y ydot th thdot")
	Fl, Ft = symbols("Fl Ft")
	state_vars = [x, xdot, y, ydot, th, thdot]
	control_vars = [Fl, Ft]

	# Given system equations
	m, g, J = 10e3, 1.6, 1e5 # kg, m/s^2. kg * m^2
	xddot = (Fl*cos(th) - Ft*sin(th)) / m
	yddot = (Fl*sin(th) + Ft*cos(th) - m*g) / m
	thddot = 4*Fl/J
	ode = [xdot, xddot, ydot, yddot, thdot, thddot]

	t0_ = 0
	tf_ = 100
	N_ = 100
	tspan = np.linspace(t0_, tf_, N_)

	# [x, xdot, y, ydot, th, thdot]
	X_start = np.array([0.5e3, 0, 3e3, 0, 0, 0], dtype=float) # known initial state
	X_goal = np.array([0, 0, 0, 0, 0, 0], dtype=float) # desired goal state
	x_bounds = [[None,None], [None,None], [0,None], [None,None], [-np.pi/4,np.pi/4], [None,None]]
	u_bounds = [[-5e3, 5e3],[0, 40e3]]
	bounds = x_bounds + u_bounds

	# Define problem
	print("Setup")
	problem = CollocationProblem(state_vars, control_vars, ode, tspan, X_start, X_goal, colloc_method)

	# solve problem
	print("Solve")
	sol_c = problem.solve(bounds=bounds, solver='scipy')

	# evaluate solution
	problem.evaluate(ivp_method='Radau')

	# draw lander
	draw_lander(sol_c.x, sol_c.u, save_anim=False)
