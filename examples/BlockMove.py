"""

Block-move example

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify

from pydcol.Animator import draw_block
from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

if __name__ == "__main__":

	colloc_method = RADAU

	# define variables
	x, v = symbols("x v")
	u = symbols("u")
	state_vars = [x, v]
	control_vars = [u]

	# Given system equations
	ode = [v, u]

	t0_ = 0
	tf_ = 1
	N_ = 100
	tspan = np.linspace(t0_, tf_, N_)

	X_start = np.array([0, 0]) # known initial state
	X_goal = np.array([1, 0]) # desired goal state

	# bounds = [[lb_x, ub_x],[lb_v, ub_v],[lb_u, ub_u]]
	u_max = 10
	bounds = [[None,None],[None,None],[-u_max, u_max]]

	# Define problem
	print("Setup")
	problem = CollocationProblem(state_vars, control_vars, ode, tspan, X_start, X_goal, colloc_method)

	# solve problem
	print("Solve")
	sol_c = problem.solve(bounds=bounds, solver='scipy')

	# evaluate solution
	problem.evaluate()

	# animate solution
	draw_block(sol_c.x, save_anim=False)
