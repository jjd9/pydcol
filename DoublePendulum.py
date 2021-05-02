"""

Double pendulum example

Authors: John D'Angelo, Shreyas Sudhaman

"""

import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify

from pydcol.Animator import draw_double_pendulum
from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

if __name__ == "__main__":

	colloc_method = HERM

	print("Initialize")
	# physical parameters
	l1 = 2.0
	l2 = 2.0
	m1 = 3.0
	m2 = 3.0
	g = 9.81

	# define variables
	theta, th_dot, phi, phi_dot = symbols("theta th_dot phi phi_dot")
	tau = symbols("tau")
	state_vars = [theta, th_dot, phi, phi_dot]
	control_vars = [tau]

	# Given system equations
	ode = [th_dot,
		(l2*(g*m1*sin(theta) + g*m2*sin(theta) - l2*m2*sin(phi - theta)*phi_dot**2) - (g*l2*m2*sin(phi) +
													  l1*l2*m2*sin(phi - theta)*th_dot**2 - tau)*cos(phi - theta))/(l1*l2*(-m1 + m2*cos(phi - theta)**2 - m2)),
		phi_dot,
		(-l2*m2*(g*m1*sin(theta) + g*m2*sin(theta) - l2*m2*sin(phi - theta)*phi_dot**2)*cos(phi - theta) + (m1 + m2)
		 * (g*l2*m2*sin(phi) + l1*l2*m2*sin(phi - theta)*th_dot**2 - tau))/(l2**2*m2*(-m1 + m2*cos(phi - theta)**2 - m2))
		]

	tf_bound = [2,6]
	N_ = 100

	X_start = np.array([0, 0, 0, 0], dtype=float) # arbitrary goal state
	X_goal = np.array([np.pi, 0, np.pi, 0], dtype=float) # arbitrary goal state

	# bounds
	u_max = 100
	bounds = [[-2*np.pi,2*np.pi],[None, None],[-2*np.pi,2*np.pi],[None, None],[-u_max,u_max]]

	# Define problem
	problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tf_bound, N_, colloc_method)

	# solve problem
	sol_c = problem.solve(bounds=bounds, solver='ipopt')

	# evaluate solution
	problem.evaluate()

	# animate solution
	draw_double_pendulum(sol_c.x, [l1, l2, m1, m2, g])
