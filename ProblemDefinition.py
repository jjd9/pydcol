# third paty imports
import numpy as np
from scipy.integrate._ivp.common import num_jac
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import solve_ivp
from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise
import matplotlib.pyplot as plt

# pydcol imports
from Objective import Objective
from EqualityConstraints import EqualityConstraints
from CollocMethods import *
from Solution import Solution

class CollocationProblem:

	def __init__(self, 
				state_vars, 
				control_vars, 
				ode, 
				X_start, 
				X_goal, 
				tspan,
				colloc_method):

		self.ode = ode
		self.state_vars = state_vars
		self.control_vars = control_vars
		self.ode_fun = lambdify(self.state_vars+self.control_vars, Matrix(self.ode), 'numpy')
		self.colloc_method = colloc_method

		self.X_start = X_start
		self.X_goal = X_goal

		# Get variable dimensions
		self.tspan = tspan
		self.N = tspan.size
		self.X_dim = len(state_vars)
		self.U_dim = len(control_vars)
		self.all_vars = state_vars + control_vars

		self.h = Symbol("h")  # symbolic time step
		self._h = tspan[1:] - tspan[:-1]  # time steps

		# Create a set of "prev" variables for accessing values at previous time step
		self.prev_all_vars = [Symbol(str(var)+"_prev") for var in self.all_vars]

		self.mid_all_vars = [Symbol(str(var)+"_mid") for var in self.all_vars]

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

	def solve(self, x0=None, bounds=None, umax=0.0):
		
		self.is_solved = False
		_bounds = bounds * self.N

		if x0 is None:
			u_mid = 0.1
			# Initialize optimization variables
			x0 = [self.X_start.tolist() + [u_mid]]
			for i in range(self.N - 1):
				xnew = self.X_start + (self.X_goal - self.X_start) * i / self.N
				x0.append(xnew.tolist() + [u_mid])
			x0 = np.array(x0).ravel()

		# Problem constraints
		constr_eq = NonlinearConstraint(self.equality_constr.eval,
										lb=0,
										ub=0,
										jac=self.equality_constr.jac,
										hess=self.equality_constr.hess)

		# Solve Problem
		sol_opt = minimize(self.objective.eval,
						x0,
						method="trust-constr",
						jac=self.objective.jac,
						hess=self.objective.hess,
						constraints=(constr_eq),
						bounds=_bounds,
						options={'sparse_jacobian': True})
		print("Done")
		if sol_opt.success:
			print("Success :-)")
		else:
			print("Failure :-(")

		print("Constraint violation: ", sol_opt.constr_violation)
		print("Iterations: ", sol_opt.niter)

		self.is_solved = sol_opt.success

		# convert scipy solution to our format
		self.sol_c = Solution(sol_opt, self.colloc_method, (self.N, self.X_dim, self.U_dim), self.tspan)

		return self.sol_c

	def evaluate(self):
		tspan = self.sol_c.t
		X = self.sol_c.x
		U = self.sol_c.u
		
		def system_eqs(t, x_t):
			U_t = np.array([self.sol_c.u_t(t)], dtype= np.float)
			return self.ode_fun(*x_t, *U_t).ravel()

		eval_tspan = np.linspace(tspan[0],tspan[-1],100)
		sol_ivp = solve_ivp(system_eqs, [tspan[0],tspan[-1]], self.X_start, method='RK45', t_eval=eval_tspan)

		colors = ['k', 'g', 'b', 'r', 'c', 'm', 'y']

		fig, axs = plt.subplots(2, 1)
		axs[0].set_title("Collocation Points vs. Integration Results")
		for i in range(self.X_dim):
			axs[0].plot(tspan, X[:,i],'o',color=colors[i])
			axs[0].plot(sol_ivp.t, sol_ivp.y[i,:],color=colors[i])
		axs[0].set_ylabel("State Variables")
		axs[0].plot([], [],'o',color='k',label='Colloc solution')
		axs[0].plot([], [],color='k',label='IVP solution')
		axs[0].legend()

		U_t = np.array(self.sol_c.u_t(sol_ivp.t)).reshape(-1, self.U_dim)
		for j in range(self.U_dim):
			axs[1].plot(tspan, U[:,j],'o',color=colors[j])
			axs[1].plot(sol_ivp.t, U_t[:,j],color=colors[j])
		axs[1].set_ylabel("Control Variables")
		axs[1].set_xlabel("Time [s]")

		plt.show()