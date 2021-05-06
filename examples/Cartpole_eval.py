"""

Cartpole example with error analysis

Authors: John D'Angelo, Shreyas Sudhaman

"""

import sys
sys.path.insert(0, '..')

import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify
from copy import deepcopy

from pydcol.Animator import draw_cartpole
from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

import matplotlib.pyplot as plt

if __name__ == "__main__":
	print("Initialize")

	# CHOOSE HOW TO EVALUATE THE ERROR:
	# 0 == state error
	# 1 == control error
	# 2 == objective error
	analysis_type = 1

	# collocation type
	colloc_methods = [EB, TRAP, HERM, RADAU]

	# physical parameters
	l = 3.0
	m1 = 3.0 # cart mass
	m2 = 0.5 # mass at end of pole
	g = 9.81

	# define variables
	q1, q2, q1_dot, q2_dot = symbols("q1 q2 q1_dot q2_dot")
	u = symbols("u")
	state_vars = [q1, q2, q1_dot, q2_dot]
	control_vars = [u]

	# Given system equations
	q1_d2dot = (l*m2*sin(q2)*q2_dot**2 + u + m2*g*cos(q2)*sin(q2))/(m1 + m2*(1-cos(q2)**2))
	q2_d2dot = - (l*m2*cos(q2)*sin(q2)*q2_dot**2 + u*cos(q2) + (m1+m2)*g*sin(q2))/(l*m1 + l*m2*(1-cos(q2)**2))
	ode = [q1_dot, q2_dot, q1_d2dot, q2_d2dot]

	dist = -4.0 # distance traveled during swing-up maneuver

	X_start = np.array([0, 0, 0, 0]) # arbitrary goal state
	X_goal = np.array([dist, np.pi, 0, 0]) # arbitrary goal state

	t0_ = 0
	tf_ = 5
	N_ = 10 # starting with too few segments results in numerical instability, 10 was sufficient

	# bounds
	u_max = 100
	dist_min, dist_max = -10, 10
	bounds = [[dist_min, dist_max],[-2*np.pi,2*np.pi],[-100,100],[-100,100],[-u_max,u_max]]

	error = {}
	for colloc_method in colloc_methods:
		error[colloc_method] = []

	for colloc_method in colloc_methods:
		tspan = np.linspace(t0_, tf_, N_)

		last_sol = None
		segments = []
		for i in range(7):
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
				segments.append(tspan.size)

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
