"""
Equality constraint definition.

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

# third party imports
import IPython
import numpy as np
from scipy.sparse import coo_matrix
from symengine import Lambdify
from sympy import Matrix

# pydcol imports
from .SymUtils import fast_jac, fast_half_hess

class EqualityConstraints:
	def __init__(self, parent, C_eq):
		self.N = parent.N
		self.Ntilde = parent.Ntilde
		
		self.colloc_method = parent.colloc_method

		self.X_dim = parent.X_dim
		self.U_dim = parent.U_dim

		self.tf = parent.tf

		self.X_start = parent.X_start
		self.X_goal = parent.X_goal

		all_vars = parent.all_vars
		prev_all_vars = parent.prev_all_vars
		mid_all_vars = parent.mid_all_vars

		self.ceq_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.tf], C_eq, order='F')

		# jacobian matrix ("jac")
		ceq_jac = Matrix(fast_jac(C_eq, prev_all_vars+all_vars + mid_all_vars+[self.tf])).T
		self.ceq_jac_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.tf], ceq_jac, order='F')

		# Hessian Matrix ("hess")
		# lagrange multipliers
		self.ncon = len(C_eq)
		lamb = Matrix(["lambda" + str(i) for i in range(self.ncon)]).reshape(self.ncon, 1)
		ceq_hess = Matrix(fast_half_hess((C_eq.T * lamb)[0], prev_all_vars + all_vars + mid_all_vars+[self.tf]))
		self.ceq_hess_lamb = Lambdify(prev_all_vars+mid_all_vars+all_vars+list(lamb)+[self.tf], ceq_hess, order='F', cse=True, backend='llvm')

		# linear if symbolic hessian is all zero
		if len(ceq_hess.free_symbols) == 0:
			self.is_linear = True
		else:
			self.is_linear = False

		import numdifftools as nd
		import matplotlib.pyplot as plt

		x0 = np.ones(self.Ntilde * (self.X_dim + self.U_dim) + 1)
		ncon = self.eval(x0).size
		lagrange = np.ones(ncon)

		self.jac_sparse_indices = self.jac(x0, return_sparse_indices=True)
		self.jac_shape = (ncon, x0.size)
		self.jac_size = len(self.jac_sparse_indices[0])
		self.jac_dict = dict()
		for i in range(self.jac_size):
			key = (self.jac_sparse_indices[0][i],self.jac_sparse_indices[1][i])
			self.jac_dict[key] = i

		self.hess_sparse_indices = self.hess(x0, lagrange, return_sparse_indices=True)
		self.hess_shape = (x0.size, x0.size)
		self.hess_size = len(self.hess_sparse_indices[0])
		self.hess_dict = dict()
		for i in range(self.hess_size):
			key = (self.hess_sparse_indices[0][i],self.hess_sparse_indices[1][i])
			self.hess_dict[key] = i


	def eval(self, arg):
		"""
		Evaluate equality constraints for given value of optimization variable.

		Parameters
		----------
		arg -- optimization variables as 1-D numpy array. 

		Returns
		-------
		vector of equality constraint residuals as 1-D numpy array. 
		"""

		arg_x = arg[:-1]
		arg_tf = arg[-1]
		_tf = arg_tf * np.ones((self.N-1,1))

		if self.N == self.Ntilde:
			V = arg_x.reshape(self.N, self.X_dim+self.U_dim)
			_X = V[:,:self.X_dim]
			_in = np.hstack((V[:-1,:], V[1:,:], _tf))
		else:
			V = arg_x[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg_x[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_X = V[:,:self.X_dim]
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:], _tf))
		_out = self.ceq_lambda(_in.T).T.ravel()
		initial_constr = (_X[0,:] - self.X_start).ravel()
		terminal_constr = (_X[-1,:] - self.X_goal).ravel()
		return np.hstack((_out, initial_constr, terminal_constr))

	def jac(self, arg: np.array, return_sparse_indices: bool = False):
		"""
		Evaluate jacobian of equality constraints for given value of optimization variable.

		Parameters
		----------
		arg -- optimization variables as 1-D numpy array. 
		return_sparse_indices -- if True return a tuple of the row, column indices of the non-zero entries of the jacobian matrix. if False, return the actual jacobian.

		Returns
		-------
		jacobian matrix of equality constraint residuals as a scipy sparse matrix (specifically a csr_matrix). 
		OR
		tuple of (row,col) indices of non-zero elements of jacobian matrix
		"""

		arg_x = arg[:-1]
		arg_tf = arg[-1]
		_tf = arg_tf * np.ones((self.N-1, 1))

		if self.N == self.Ntilde:
			V = arg_x.reshape(self.Ntilde, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], V[1:,:], _tf))
		else:
			V = arg_x[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg_x[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:], _tf))

		J = self.ceq_jac_lambda(_in.T)

		# jac should be Number constraints x Number optimization variables
		Opt_dim = (self.X_dim + self.U_dim)
		Ceq_dim = self.ncon		
		jac_shape = (Ceq_dim * (self.N-1) + 2 * self.X_dim, Opt_dim * self.Ntilde + 1)

		# used for determining nonzero elements of jacobian
		if return_sparse_indices:
			rows = []
			cols = []
			for i in range(self.N-1):
				for j in range(i*Ceq_dim, i*Ceq_dim + Ceq_dim):
					for k in range(i*Opt_dim, (i+1)*Opt_dim + Opt_dim):
						rows.append(j)
						cols.append(k)
					rows.append(j)
					cols.append(jac_shape[1]-1)
				if self.N != self.Ntilde:
					for j in range(i*Ceq_dim, i*Ceq_dim + Ceq_dim):
						for k in range((i + self.N)*Opt_dim, (i + self.N)*Opt_dim + Opt_dim):
							rows.append(j)
							cols.append(k)
			# initial and terminal constraint gradients are easy
			rows += np.arange(Ceq_dim * (self.N-1), Ceq_dim * (self.N-1) + self.X_dim).tolist()
			rows += np.arange(Ceq_dim * (self.N-1) + self.X_dim, jac_shape[0]).tolist()
			cols += np.arange(0, self.X_dim).tolist()
			if self.N != self.Ntilde:
				cols += np.arange(jac_shape[1]-1-(self.N-1)*(self.X_dim+self.U_dim)-(self.X_dim+self.U_dim), jac_shape[1]-1-(self.N-1)*(self.X_dim+self.U_dim)-self.U_dim).tolist()
			else:
				cols += np.arange(jac_shape[1]-1-(self.X_dim+self.U_dim), jac_shape[1]-1-self.U_dim).tolist()
			return rows, cols
		else:
			jac = np.zeros(self.jac_size, dtype=float)

			for i in range(self.N-1):
				for j in range(i*Ceq_dim, i*Ceq_dim + Ceq_dim):
					for k in range(i*Opt_dim, (i+1)*Opt_dim + Opt_dim):
						jac[self.jac_dict[(j,k)]]+=J[k-i*Opt_dim,j-i*Ceq_dim,i]
					jac[self.jac_dict[(j,jac_shape[1]-1)]]+=J[-1,j-i*Ceq_dim,i]
				if self.N != self.Ntilde:
					for j in range(i*Ceq_dim, i*Ceq_dim + Ceq_dim):
						for k in range((i + self.N)*Opt_dim, (i + self.N)*Opt_dim + Opt_dim):
							jac[self.jac_dict[(j,k)]]+=J[2*Opt_dim + k-(i + self.N)*Opt_dim,j-i*Ceq_dim,i]

			for j in range(Ceq_dim * (self.N-1), Ceq_dim * (self.N-1) + self.X_dim):
				jac[self.jac_dict[(j,j-Ceq_dim * (self.N-1))]] = 1.0

			if self.N != self.Ntilde:
				col_range = (jac_shape[1]-1-(self.N-1)*(self.X_dim+self.U_dim)-(self.X_dim+self.U_dim), jac_shape[1]-1-(self.N-1)*(self.X_dim+self.U_dim)-self.U_dim)
			else:
				col_range = (jac_shape[1]-1-(self.X_dim+self.U_dim), jac_shape[1]-1-self.U_dim)

			row_range = (Ceq_dim * (self.N-1) + self.X_dim, jac_shape[0])

			for j in range(self.X_dim):
				jac[self.jac_dict[(j+row_range[0],j+col_range[0])]] = 1.0

			return coo_matrix((jac, self.jac_sparse_indices),shape=self.jac_shape)


	def hess(self, arg, arg_v, return_sparse_indices=False):
		"""
		Evaluate hessian of equality constraints for given value of optimization variable.

		Parameters
		----------
		arg -- optimization variables as 1-D numpy array. 
		return_sparse_indices -- if True return a tuple of the row, column indices of the non-zero entries of the hessian matrix. if False, return the actual hessian.

		Returns
		-------
		hessian matrix of equality constraint residuals as a scipy sparse matrix (specifically a lil_matrix). 
		OR
		tuple of (row,col) indices of non-zero elements of hessian matrix
		"""

		arg_x = arg[:-1]
		arg_tf = arg[-1]
		_tf = arg_tf * np.ones((self.N-1, 1))

		hess_shape = (arg.size, arg.size)
		if self.is_linear:
			if return_sparse_indices:
				return [], []
			else:
				return coo_matrix(hess_shape, dtype=np.float)
		else:
			if self.N == self.Ntilde:
				V = arg_x.reshape(self.N, self.X_dim+self.U_dim)
				_L = arg_v[:-2*self.X_dim].reshape(self.N-1, self.ncon)
				_in = np.hstack((V[:-1,:], V[1:,:], _L, _tf))
			else:
				V = arg_x[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
				Vmid = arg_x[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
				_L = arg_v[:-2*self.X_dim].reshape(self.N-1, self.ncon)
				_in = np.hstack((V[:-1,:], Vmid, V[1:,:], _L, _tf))

			H = self.ceq_hess_lamb(_in.T)

			# used for determining nonzero elements of hessian
			Sys_dim = (self.X_dim + self.U_dim)

			if self.N != self.Ntilde:
				hdim = 3*Sys_dim+1
			else:
				hdim = 2*Sys_dim+1

			if return_sparse_indices:
				idx = set()
				for i in range(self.N-1):
					for j in range(hdim):
						for k in range(j, hdim):							
							if j < 2*Sys_dim and k < 2*Sys_dim: # A
								idx.add((i*Sys_dim+j, i*Sys_dim+k))
								idx.add((i*Sys_dim+k, i*Sys_dim+j))
							elif self.N != self.Ntilde and j >= 2*Sys_dim and j < 3*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # B
								idx.add(((i + self.N)*Sys_dim+j-2*Sys_dim, (i + self.N)*Sys_dim+k-2*Sys_dim))
								idx.add(((i + self.N)*Sys_dim+k-2*Sys_dim, (i + self.N)*Sys_dim+j-2*Sys_dim))
							elif self.N != self.Ntilde and j < 2*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # C == D
								idx.add((i*Sys_dim+j, (i + self.N)*Sys_dim+k-2*Sys_dim))
								idx.add(((i + self.N)*Sys_dim+k-2*Sys_dim, i*Sys_dim+j))
							elif j < 2*Sys_dim and k == hdim-1: # E==F
								idx.add((i*Sys_dim+j, arg.size-1))
								idx.add((arg.size-1, i*Sys_dim+j))
							elif self.N != self.Ntilde and j >= 2*Sys_dim and k == hdim-1: # E==F
								idx.add(((i + self.N)*Sys_dim+j-2*Sys_dim, arg.size-1))
								idx.add((arg.size-1, (i + self.N)*Sys_dim+j-2*Sys_dim))
							else:
								idx.add((arg.size-1,arg.size-1))
				idx = np.array(list(idx))
				return idx[:,0], idx[:,1]
			else:
				hess = np.zeros(self.hess_size, dtype=np.float)

				for i in range(self.N-1):
					Htemp = H[:,:,i] + H[:,:,i].T
					for j in range(hdim):
						for k in range(j, hdim):
							if j < 2*Sys_dim and k < 2*Sys_dim: # A
								hess[self.hess_dict[(i*Sys_dim+j, i*Sys_dim+k)]]+=Htemp[j,k]
								hess[self.hess_dict[(i*Sys_dim+k, i*Sys_dim+j)]]+=Htemp[k,j]
							elif self.N != self.Ntilde and j >= 2*Sys_dim and j < 3*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # B
								hess[self.hess_dict[((i + self.N)*Sys_dim+j-2*Sys_dim, (i + self.N)*Sys_dim+k-2*Sys_dim)]]+=Htemp[j,k]
								hess[self.hess_dict[((i + self.N)*Sys_dim+k-2*Sys_dim, (i + self.N)*Sys_dim+j-2*Sys_dim)]]+=Htemp[k,j]
							elif self.N != self.Ntilde and j < 2*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # C == D
								hess[self.hess_dict[(i*Sys_dim+j, (i + self.N)*Sys_dim+k-2*Sys_dim)]]+=Htemp[j,k]
								hess[self.hess_dict[((i + self.N)*Sys_dim+k-2*Sys_dim, i*Sys_dim+j)]]+=Htemp[k,j]
							elif j < 2*Sys_dim and k == hdim-1: # E==F
								hess[self.hess_dict[(i*Sys_dim+j, arg.size-1)]]+=Htemp[j,k]
								hess[self.hess_dict[(arg.size-1, i*Sys_dim+j)]]+=Htemp[k,j]
							elif self.N != self.Ntilde and j >= 2*Sys_dim and k == hdim-1: # E==F
								hess[self.hess_dict[((i + self.N)*Sys_dim+j-2*Sys_dim, arg.size-1)]]+=Htemp[j,k]
								hess[self.hess_dict[(arg.size-1, (i + self.N)*Sys_dim+j-2*Sys_dim)]]+=Htemp[k,j]
							else:
								hess[self.hess_dict[(arg.size-1,arg.size-1)]]+=Htemp[j,k]
				return coo_matrix((hess, self.hess_sparse_indices),shape=self.hess_shape)
