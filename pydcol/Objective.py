"""

Objective function definition.

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

# third party imports
import numpy as np
from scipy.sparse import csr_matrix
from symengine import Lambdify
from sympy import Matrix, hessian
from typing import Union

# pydcol imports
from .SymUtils import fast_jac, fast_half_hess

class Objective:
	def __init__(self, parent, Obj):
		self.N = parent.N
		self.Ntilde = parent.Ntilde
		self.h = parent.h
		self._h = parent._h.copy()

		self.colloc_method = parent.colloc_method

		self.X_dim = parent.X_dim
		self.U_dim = parent.U_dim

		all_vars = parent.all_vars
		mid_all_vars = parent.mid_all_vars
		prev_all_vars = parent.prev_all_vars

		if self.N != self.Ntilde:
			self.obj_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], Obj, order='F')

			# Gradient vector ("jac")
			obj_jac = Matrix(fast_jac([Obj], prev_all_vars+all_vars + mid_all_vars)).T
			self.obj_jac_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], obj_jac, order='F')

			# hessian matrix ("hess")
			obj_hess = Matrix(fast_half_hess(Obj, prev_all_vars+all_vars + mid_all_vars)).T
			self.obj_hess_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], obj_hess, order='F')
		else:
			self._h = np.hstack((self._h[0],self._h))
			self.obj_lambda = Lambdify(all_vars+[self.h], Obj, order='F')

			# Gradient vector ("jac")
			obj_jac = Matrix([Obj]).jacobian(all_vars)
			self.obj_jac_lambda = Lambdify(all_vars+[self.h], obj_jac, order='F')

			# hessian matrix ("hess")
			obj_hess = hessian(Obj, all_vars)
			self.obj_hess_lambda = Lambdify(all_vars+[self.h], obj_hess, order='F')

		x0 = np.ones(self.Ntilde * (self.X_dim + self.U_dim))
		self.hess_sparse_indices = self.hess(x0, return_sparse_indices=True)        

		self.hess_shape = (x0.size, x0.size)
		self.hess_size = len(self.hess_sparse_indices[0])
		self.hess_dict = dict()
		for i in range(self.hess_size):
			key = (self.hess_sparse_indices[0][i],self.hess_sparse_indices[1][i])
			self.hess_dict[key] = i

	# create callback for scipy
	def eval(self, arg: np.array)->float:
		"""
		Evaluate objective function for given value of optimization variable.

		Parameters
		----------
		arg -- optimization variables as 1-D numpy array. 

		Returns
		-------
		scalar objective value.
		"""

		if self.N != self.Ntilde:
			V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))
		else:
			_in = np.hstack((arg.reshape(self.Ntilde, self.X_dim+self.U_dim),self._h.reshape(-1,1))) 

		return self.obj_lambda(_in.T).sum()

	def jac(self, arg: np.array)->np.array:
		"""
		Evaluate gradient vector of objective function for given value of optimization variable.

		Parameters
		----------
		arg -- optimization variables as 1-D numpy array. 

		Returns
		-------
		gradient vector of object function as 1-D numpy array.
		"""

		if self.N != self.Ntilde:
			V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))
			J = self.obj_jac_lambda(_in.T).squeeze()
			SysDim = self.X_dim + self.U_dim
			jac = np.zeros(self.Ntilde * SysDim)
			for i in range(self.N-1):
				jac[i*SysDim:(i+1)*SysDim+SysDim] += J[:SysDim*2,i]
				jac[(i+self.N)*SysDim:(i+self.N)*SysDim+SysDim] += J[SysDim*2:,i]
		else:
			_in = np.hstack((arg.reshape(self.Ntilde, self.X_dim+self.U_dim),self._h.reshape(-1,1))) 
			jac = self.obj_jac_lambda(_in.T).squeeze().T.ravel()

		return jac

	def hess(self, arg: np.array, return_sparse_indices: bool = False)->Union[tuple, csr_matrix]:
		"""
		Evaluate gradient vector of objective function for given value of optimization variable.

		Parameters
		----------
		arg -- optimization variables as 1-D numpy array. 
		return_sparse_indices -- if True return a tuple of the row, column indices of the non-zero entries of the hessian matrix. if False, return the actual hessian.

		Returns
		-------
		hessian matrix of object function as a sparse numpy matrix (lil_matrix).
		OR
		tuple of (row,col) indices of non-zero elements of hessian matrix
		"""

		Sys_dim = self.X_dim + self.U_dim
		Opt_dim = Sys_dim * self.Ntilde
		if self.N != self.Ntilde:
			V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))

			hess_block = self.obj_hess_lambda(_in.T) + 1e-9

			# used for determining nonzero elements of hessian
			if return_sparse_indices:
				idx = set()
				for i in range(self.N-1):
					for j in range(2*Sys_dim):
						for k in range(2*Sys_dim):
							idx.add((i*Sys_dim+j, i*Sys_dim+k))
					for j in range(Sys_dim):
						for k in range(Sys_dim):
							idx.add(((i + self.N)*Sys_dim+j, (i + self.N)*Sys_dim+k))
				idx = np.array(list(idx))
				return idx[:,0], idx[:,1]
			else:
				hess = np.zeros(self.hess_size, dtype=float)
				for i in range(self.N-1):
					Htemp = hess_block[:,:,i] + hess_block[:,:,i].T
					for j in range(2*Sys_dim):
						for k in range(2*Sys_dim):
							hess[self.hess_dict[(i*Sys_dim+j, i*Sys_dim+k)]]+=Htemp[j,k]
					for j in range(Sys_dim):
						for k in range(Sys_dim):
							hess[self.hess_dict[((i + self.N)*Sys_dim+j, (i + self.N)*Sys_dim+k)]]+=Htemp[2*Sys_dim+j,2*Sys_dim+k]
				return csr_matrix((hess, self.hess_sparse_indices), shape = self.hess_shape)
		else:
			_in = np.hstack((arg.reshape(self.Ntilde, self.X_dim+self.U_dim),self._h.reshape(-1,1))) 
			hess_block = self.obj_hess_lambda(_in.T) + 1e-9
			# used for determining nonzero elements of hessian
			if return_sparse_indices:
				rows = []
				cols = []
				for i in range(self.N):
					for j in range(i*Sys_dim, i*Sys_dim + Sys_dim):
						for k in range(i*Sys_dim, i*Sys_dim + Sys_dim):
							rows.append(j)
							cols.append(k)
				return rows, cols
			else:
				return csr_matrix((hess_block.ravel(), self.hess_sparse_indices), shape = (Opt_dim, Opt_dim))