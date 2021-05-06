"""

Block-move example with error analysis.

Authors: John D'Angelo, Shreyas Sudhaman

"""

import sys
sys.path.insert(0, '..')

from copy import deepcopy
import numpy as np
from sympy import symbols

from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

import matplotlib.pyplot as plt

if __name__ == "__main__":

	# CHOOSE HOW TO EVALUATE THE ERROR:
	# 0 == state error
	# 1 == control error
	# 2 == objective error
	analysis_type = 1

	# set of collocation methods to analyze
	colloc_methods = [EB, TRAP, HERM, RADAU]

	# define variables
	x, v = symbols("x v")
	u = symbols("u")
	state_vars = [x, v]
	control_vars = [u]

	# Given system equations
	ode = [v, u]

	X_start = np.array([0, 0]) # arbitrary goal state
	X_goal = np.array([1, 0]) # arbitrary goal state
	# bounds = [[lb_x, ub_x],[lb_v, ub_v],[lb_u, ub_u]]
	u_max = 10
	bounds = [[None,None],[None,None],[-u_max, u_max]]

	error = {}
	for colloc_method in colloc_methods:
		error[colloc_method] = []

	for colloc_method in colloc_methods:
		t0_ = 0
		tf_ = 1
		N_ = 3
		tspan = np.linspace(t0_, tf_, N_)

		last_sol = None
		segments = []
		for i in range(10):
			# Define problem
			problem = CollocationProblem(state_vars, control_vars, ode, tspan, X_start, X_goal, colloc_method)

			# solve problem
			print("Start solve")
			sol_c = problem.solve(bounds=bounds, solver='scipy')
			if last_sol is not None:
				if analysis_type == 0: # State
					prev_points = last_sol.x_endpts
					cur_points = sol_c.x_endpts[::2,:]
					err = np.linalg.norm(prev_points[1:-1,:] - cur_points[1:-1,:],axis=1).mean()
				elif analysis_type == 1: # Control
					prev_points = last_sol.u_endpts
					cur_points = sol_c.u_endpts[::2,:]
					err = np.linalg.norm(prev_points[1:,:] - cur_points[1:,:],axis=1).mean()
				elif analysis_type == 2: # Objective
					prev_points = last_sol.obj
					cur_points = sol_c.obj
					err = np.abs(prev_points - cur_points)
				error[colloc_method].append(err)
				segments.append(len(tspan))

			last_sol = deepcopy(sol_c)

			# divide each segment of time by 2
			new_tspan = [tspan[0]]
			for j in range(1,tspan.size):
				seg_length = tspan[j] - tspan[j-1]
				new_tspan.append(new_tspan[-1] + seg_length / 2.0)
				new_tspan.append(new_tspan[-1] + seg_length / 2.0)
			tspan = np.array(new_tspan)

	# Plot results
	fig, ax = plt.subplots()
	for colloc_method in error:
		name = METHOD_NAMES[colloc_method]
		ax.loglog(segments, error[colloc_method], label=name)
		temp_error = np.array(error[colloc_method])
		print(name, ": ", np.log(np.abs(temp_error[:-1]/temp_error[1:]))/np.log(2))
	ax.set_xlabel("Number of Segments")
	ax.set_ylabel("Error, |X(i) - X(i-1)|")
	ax.grid()
	ax.legend()
	plt.show()
