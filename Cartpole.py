"""

Cartpole example

Authors: John D'Angelo, Shreyas Sudhaman

"""

import numpy as np
from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify

from pydcol.Animator import draw_cartpole
from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem

if __name__ == "__main__":
	print("Initialize")

	# collocation type
	colloc_method = HERM

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

	t0_ = 0
	tf_ = 5
	N_ = 50

	dist = -4.0 # distance traveled during swing-up maneuver

	X_start = np.array([0, 0, 0, 0]) # arbitrary goal state
	X_goal = np.array([dist, np.pi, 0, 0]) # arbitrary goal state

	# bounds
	u_max = 100
	dist_min, dist_max = -10, 10
	bounds = [[dist_min, dist_max],[-2*np.pi,2*np.pi],[-100,100],[-100,100],[-u_max,u_max]]
	tspan = np.linspace(t0_, tf_, N_)

	# Define problem
	problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tspan, colloc_method)

	# solve problem
	sol_c = problem.solve(bounds=bounds, solver='scipy')

	# evaluate solution
	problem.evaluate(ivp_method='Radau')

	# animate solution
	draw_cartpole(sol_c.x, [l, m1, m2, g])
