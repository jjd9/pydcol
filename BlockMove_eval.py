"""

Block-move example

Authors: John D'Angelo, Shreyas Sudhaman

"""

from copy import deepcopy
import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify

from pydcol.Animator import draw_block
from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

import matplotlib.pyplot as plt

if __name__ == "__main__":

	colloc_method = TRAP

	# define variables
	x, v = symbols("x v")
	u = symbols("u")
	state_vars = [x, v]
	control_vars = [u]

	# Given system equations
	ode = [v, u]

	X_start = np.array([0, 0]) # arbitrary goal state
	X_goal = np.array([10, 0]) # arbitrary goal state
	# bounds = [[lb_x, ub_x],[lb_v, ub_v],[lb_u, ub_u]]
	u_max = 10
	bounds = [[None,None],[None,None],[-u_max, u_max]]

	t0_ = 0
	tf_ = 5
	N_ = 3
	tspan = np.linspace(t0_, tf_, N_)

	obj = []
	error = []
	last_sol = None

	for i in range(20):
		# divide each segment of time by 2
		new_tspan = [tspan[0]]
		for j in range(1,tspan.size):
			seg_length = tspan[j] - tspan[j-1]
			new_tspan.append(new_tspan[-1] + seg_length / 2.0)
			new_tspan.append(new_tspan[-1] + seg_length / 2.0)
		tspan = np.array(new_tspan)

		# Define problem
		problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tspan, colloc_method)

		# solve problem
		print("Start solve")
		sol_c = problem.solve(umax=u_max, bounds=bounds, solver='scipy')
		obj.append(sol_c.obj)
		if last_sol is not None:
			prev_points = last_sol.x
			cur_points = sol_c.x[::2,:]
			err = np.linalg.norm(cur_points - prev_points)
			error.append(err)
			print("Error: ", err)

		last_sol = deepcopy(sol_c)

	plt.plot(error)
	plt.show()
	error = np.array(error)
	print(-np.log(np.abs(error[:-1] - error[1:]))/np.log(2))
	# obj = np.array(obj)
	# print(-np.log(np.abs(obj[:-1] - obj[1:]))/np.log(2))

	# evaluate solution
	problem.evaluate()

	# # animate solution
	# draw_block(sol_c.x)
