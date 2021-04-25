"""

Block-move example

Authors: John D'Angelo, Shreyas Sudhaman

"""

import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify

from Animator import draw_block
from CollocMethods import *
from ProblemDefinition import CollocationProblem

if __name__ == "__main__":

	colloc_method = TRAP

	# define variables
	x, v = symbols("x v")
	u = symbols("u")
	state_vars = [x, v]
	control_vars = [u]

	# Given system equations
	ode = [v, u]

	t0_ = 0
	tf_ = 5
	N_ = 10
	tspan = np.linspace(t0_, tf_, N_)

	X_start = np.array([0, 0]) # arbitrary goal state
	X_goal = np.array([10, 0]) # arbitrary goal state

	# bounds = [[lb_x, ub_x],[lb_v, ub_v],[lb_u, ub_u]]
	u_max = 10
	bounds = [[None,None],[None,None],[-u_max, u_max]]

	# Define problem
	problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tspan, colloc_method)

	# solve problem
	sol_c = problem.solve(umax=u_max, bounds=bounds)

	# evaluate solution
	problem.evaluate()

	# animate solution
	draw_block(sol_c.x)
