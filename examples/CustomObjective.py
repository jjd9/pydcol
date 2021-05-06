"""

Optimal control problem with custom objective and free terminal condition.

Authors: John D'Angelo, Shreyas Sudhaman

"""

import sys
sys.path.insert(0, '..')

import numpy as np
from sympy import symbols

from pydcol.CollocMethods import *
from pydcol.ProblemDefinition import CollocationProblem
from pydcol.Objective import CustomObjective

if __name__ == "__main__":
	# collocation type
	colloc_method = EF

	# define variables
	y1, y2 = symbols("y1 y2")
	u = symbols("u")
	state_vars = [y1, y2]
	control_vars = [u]

	# Given system equations
	ode = [0.5*y1 + u, y1**2 + 0.5*u**2]

	t0_ = 0
	tf_ = 1
	N_ = 10

	X_start = np.array([1, 0]) # known initial state
	# free terminal state

	# bounds
	bounds = [[None, None],[None, None],[None, None]]
	tspan = np.linspace(t0_, tf_, N_)

	# Custom objective
	def my_eval(arg):
		# we want to minimize y2(1)
		return arg[N_ * 3 - 2]
	def my_jac(arg):
		jac = np.zeros(arg.size, dtype=float)
		jac[N_ * 3 - 2] = 1
		return jac

	def my_hess(arg):
		return np.zeros((arg.size, arg.size), dtype=float)

	my_objective = CustomObjective()
	my_objective.eval = my_eval
	my_objective.jac = my_jac
	my_objective.hess = my_hess

	# Define problem
	print("Setup")
	problem = CollocationProblem(state_vars, control_vars, ode, tspan, X_start, colloc_method=colloc_method, custom_objective=my_objective)

	# solve problem
	print("Solve")
	sol_c = problem.solve(bounds=bounds, solver='scipy')

	# evaluate solution
	problem.evaluate(ivp_method='Radau')
