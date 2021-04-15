from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise

import numpy as np

from Objective import Objective
from EqualityConstraints import EqualityConstraints

from colloc_methods import *

class CollocationProblem:

	def __init__(self, 
				state_vars, 
				control_vars, 
				ode, 
				X_start, 
				X_goal, 
				tspan,
				colloc_method):

		self.X_start = X_start
		self.X_goal = X_goal

		# Get variable dimensions
		self.N = tspan.size
		self.X_dim = len(state_vars)
		self.U_dim = len(control_vars)
		self.all_vars = state_vars + control_vars

		self.h = Symbol("h")  # symbolic time step
		self._h = tspan[1:] - tspan[:-1]  # time steps

		# Create a set of "prev" variables for accessing values at previous time step
		self.prev_all_vars = [Symbol(str(var)+"_prev") for var in self.all_vars]

		self.prev_dict = {}
		for i in range(len(self.all_vars)):
			self.prev_dict[self.all_vars[i]] = self.prev_all_vars[i]

		# internal optimization variable maps
		self.X_sym = Symbol("X")
		self.U_sym = Symbol("U")
		self.X_prev_sym = Symbol("Xprev")
		self.U_prev_sym = Symbol("Uprev")

		X = Matrix(state_vars)
		U = Matrix(control_vars)

		# Scalar Objective
		err = X - Matrix(X_goal)
		state_error = err.multiply_elementwise(err)
		effort = U.multiply_elementwise(U)
		Obj = 0.1 * np.sum(state_error[:]) + np.sum(effort[:])

		print("Objective")
		self.objective = Objective(self, Obj)

		# Equality Constraints
		C_eq = []
		if colloc_method == TRAP:
			# Trapezoid method
			for i in range(self.X_dim):
				C_eq += [state_vars[i] - state_vars[i].subs(self.prev_dict) - 0.5 * self.h * (ode[i] + ode[i].subs(self.prev_dict))]
		elif colloc_method == HERM:
			# Hermite Simpson method
			mid_dict = {}
			for j in range(len(control_vars)):
				mid_dict[control_vars[j]] = 0.5 * (control_vars[j] + control_vars[j].subs(self.prev_dict))
			for i in range(self.X_dim):
				mid_dict[state_vars[i]] = 0.5 * (state_vars[i] + state_vars[i].subs(self.prev_dict)) + (self.h/8.0) * (ode[i].subs(self.prev_dict) - ode[i])
			for i in range(self.X_dim):
				C_eq += [state_vars[i] - state_vars[i].subs(self.prev_dict) - (self.h/6.0) * (ode[i] + 4.0 * ode[i].subs(mid_dict) + ode[i].subs(self.prev_dict))]

		print("Equality constraints")
		self.equality_constr = EqualityConstraints(self, Matrix(C_eq))
