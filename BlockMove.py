"""

Block-move example

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify

from pydcol.Animator import draw_block
from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

if __name__ == "__main__":

	colloc_method = HERM

	# define variables
	x, v = symbols("x v")
	u = symbols("u")
	state_vars = [x, v]
	control_vars = [u]

	# Given system equations
	ode = [v, u]

	N_ = 10
	tf_bound = [1, 5]

	X_start = np.array([0, 0]) # arbitrary goal state
	X_goal = np.array([1, 0]) # arbitrary goal state

	# bounds = [[lb_x, ub_x],[lb_v, ub_v],[lb_u, ub_u]]
	u_max = 10
	bounds = [[None,None],[None,None],[-u_max, u_max]]

	# Define problem
	problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tf_bound, N_, colloc_method)

	# solve problem
	sol_c = problem.solve(bounds=bounds, solver='ipopt')

	# evaluate solution
	problem.evaluate()

	# animate solution
	# draw_block(sol_c.x)
