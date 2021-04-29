# third party imports
try:
	import ipyopt
	_ipyopt_imported = True
except:
	_ipyopt_imported = False

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate._ivp.common import num_jac
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import solve_ivp
from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.core.function import BadArgumentsError
from sympy.matrices.dense import matrix_multiply_elementwise

# pydcol imports
from .Objective import Objective
from .EqualityConstraints import EqualityConstraints
from .CollocMethods import *
from .Solution import Solution

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
		self.tspan = tspan

		self.X_start = X_start
		self.X_goal = X_goal

		# Get variable dimensions
		self.N = self.tspan.size
		self.Ntilde=self.tspan.size
		self.X_dim = len(state_vars)
		self.U_dim = len(control_vars)
		self.all_vars = state_vars + control_vars

		self.h = Symbol("h")  # symbolic time step
		self._h = self.tspan[1:] - self.tspan[:-1]  # time steps

		# Create a set of "prev" and "mid" variables for accessing values at previous time step
		self.prev_all_vars = [Symbol(str(var)+"_prev") for var in self.all_vars]
		self.prev_dict = {}
		for i in range(len(self.all_vars)):
			self.prev_dict[self.all_vars[i]] = self.prev_all_vars[i]

		if self.colloc_method in MIDPOINT_METHODS:
			self.mid_all_vars = [Symbol(str(var)+"_mid") for var in self.all_vars]
			self.mid_dict = {}
			for i in range(len(self.all_vars)):
				self.mid_dict[self.all_vars[i]] = self.mid_all_vars[i]
		else:
			self.mid_all_vars = []	

		X = Matrix(state_vars)
		U = Matrix(control_vars)

		# Scalar Objective
		err = X - Matrix(X_goal)
		state_error = err.multiply_elementwise(err)
		effort = U.multiply_elementwise(U)
		if self.colloc_method in MIDPOINT_METHODS:
			Obj = (self.h/6.0) * (self.control_vars[0]**2 + 4.0 * self.control_vars[0].subs(self.mid_dict)**2 + self.control_vars[0].subs(self.prev_dict)**2)
		else:
			Obj = np.sum(effort[:])

		# Equality Constraints
		C_eq = []
		if colloc_method == TRAP:
			# Trapezoid method
			for i in range(self.X_dim):
				C_eq += [state_vars[i] - state_vars[i].subs(self.prev_dict) - 0.5 * self.h * (ode[i] + ode[i].subs(self.prev_dict))]
		elif colloc_method == EB:
			# Euler Backward method
			for i in range(self.X_dim):
				C_eq += [state_vars[i] - state_vars[i].subs(self.prev_dict) - self.h * ode[i]]
		elif colloc_method == EF:
			# Euler Forward method
			for i in range(self.X_dim):
				C_eq += [state_vars[i] - state_vars[i].subs(self.prev_dict) - self.h * ode[i].subs(self.prev_dict)]
		elif colloc_method == HERM:
			# Hermite Simpson method
			self.Ntilde=self.Ntilde*2-1 # actual number of node points due to addition of "mid" points
			for i in range(self.X_dim):
				C_eq+=[state_vars[i].subs(self.mid_dict) - 0.5 * (state_vars[i] + state_vars[i].subs(self.prev_dict)) - (self.h/8.0) * (ode[i].subs(self.prev_dict) - ode[i])]
			for i in range(self.X_dim):
				C_eq += [state_vars[i] - state_vars[i].subs(self.prev_dict) - (self.h/6.0) * (ode[i] + 4.0 * ode[i].subs(self.mid_dict) + ode[i].subs(self.prev_dict))]
		elif colloc_method == RADAU:
			# Radau 3rd order
			self.Ntilde=self.Ntilde*2-1 # actual number of node points due to addition of "mid" points
			for i in range(self.X_dim):
				C_eq+=[state_vars[i].subs(self.mid_dict) - state_vars[i].subs(self.prev_dict)-5.0/12.0*self.h*ode[i].subs(self.mid_dict)+1.0/12.0*self.h*ode[i]] #intermediate point residue
			for i in range(self.X_dim):
				C_eq+=[state_vars[i] - state_vars[i].subs(self.prev_dict)-3.0/4.0*self.h*ode[i].subs(self.mid_dict)-1.0/4.0*self.h*ode[i]] #intermediate point residue
              		

		# Compile objective and equality constraints
		print("Objective")
		self.objective = Objective(self, Obj)

		print("Equality constraints")
		self.equality_constr = EqualityConstraints(self, Matrix(C_eq))

	def solve(self, x0=None, bounds=None, umax=0.0, solver='scipy'):
		
		self.is_solved = False

		if x0 is None:
			u_mid = 0.1
			# Initialize optimization variables
			x0 = [self.X_start.tolist() + [u_mid]]
			x0_mid = []
			for i in range(self.N - 1):
				xnew = self.X_start + (self.X_goal - self.X_start) * i / self.Ntilde
				x0.append(xnew.tolist() + [u_mid])
				if self.N != self.Ntilde:
					x0_mid.append(0.5*(np.array(x0[-1]) + np.array(x0[-2])))
			x0 = np.array(x0 + x0_mid).ravel()

		if solver=='scipy':
			_bounds = bounds * self.Ntilde

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

			# convert scipy solution to our format
			self.sol_c = Solution(sol_opt, self.colloc_method, (self.N, self.Ntilde, self.X_dim, self.U_dim), self.tspan, solver)
			self.is_solved = sol_opt.success
		elif solver == "ipopt":
			if not _ipyopt_imported:
				raise(ImportError("Ipyopt could not be imported! Please use scipy solver."))			
			# setup variable bounds
			nvar = self.Ntilde * len(bounds)
			x_L = np.zeros(nvar)
			x_U = np.zeros(nvar)
			v_idx = 0
			for i in range(self.Ntilde):
				for b_pair in bounds:
					if b_pair[0] is None:
						x_L[v_idx] = -1e9
					else:
						x_L[v_idx] = b_pair[0]
					if b_pair[1] is None:
						x_U[v_idx] = 1e9
					else:
						x_U[v_idx] = b_pair[1]						
					v_idx += 1

			# setup equality constraints
			ncon = self.equality_constr.eval(x0).size
			g_L = np.zeros((ncon,))
			g_U = np.zeros((ncon,))

			# finding out which entries of the constraint jacobian and problem hessian are allways
			# nonzero. 
			ones_jac_g = self.equality_constr.jac(x0, fill=True)
			eval_jac_g_sparsity_indices = np.column_stack(np.where(ones_jac_g != 0))
			jac_g_idx = (eval_jac_g_sparsity_indices[:,0], eval_jac_g_sparsity_indices[:,1])

			lagrange = np.ones(ncon)
			ones_h = self.objective.hess(x0, fill=True) + self.equality_constr.hess(x0, lagrange, fill=True)
			eval_h_sparsity_indices = np.column_stack(np.where(ones_h != 0))
			h_idx = (eval_h_sparsity_indices[:,0], eval_h_sparsity_indices[:,1])

			def eval_grad_f(x, out):
				out[()] = self.objective.jac(x).ravel()
				return out

			def eval_g(x, out):
				out[()] = self.equality_constr.eval(x).ravel()
				return out

			def eval_jac_g(x, out):
				J = self.equality_constr.jac(x)
				out[()] = J[jac_g_idx[0], jac_g_idx[1]].ravel()
				return out

			def eval_h(x, lagrange, obj_factor, out):
				"""
				Combined hessian for the problem. Used by ipopt.
				"""
				H = self.objective.hess(x) * obj_factor + self.equality_constr.hess(x, lagrange)
				out[()] = H[h_idx[0], h_idx[1]].ravel()
				return out

			nlp = ipyopt.Problem(nvar, x_L, x_U, 
								 ncon, g_L, g_U, 
								 jac_g_idx, h_idx, 
								 self.objective.eval, eval_grad_f, 
								 eval_g, eval_jac_g, 
								 eval_h)
			nlp.set(print_level=0)
			sol_x, _, status = nlp.solve(x0)
			# convert scipy solution to our format
			self.sol_c = Solution(sol_x, self.colloc_method, (self.Ntilde, self.X_dim, self.U_dim), self.tspan, solver)
			self.is_solved = (status == 0) or (status == 1) # solver either succeeded or converged to acceptable accuracy
		else:
			raise(BadArgumentsError("Error unsupported solver!"))

		print("Done")
		if self.is_solved:
			print("Success :-)")
		else:
			print("Failure :-(")

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

		_, axs = plt.subplots(2, 1)
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