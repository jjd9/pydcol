"""

Objective function definition.

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

# third party imports
import numpy as np
from scipy.sparse import coo_matrix
from symengine import Lambdify
from sympy import Matrix, hessian

# pydcol imports
from .SymUtils import fast_jac, fast_half_hess

class Objective:
	def __init__(self, parent, Obj):
		self.N = parent.N
		self.Ntilde = parent.Ntilde
		self.tf = parent.tf

		self.colloc_method = parent.colloc_method

		self.X_dim = parent.X_dim
		self.U_dim = parent.U_dim

		all_vars = parent.all_vars
		mid_all_vars = parent.mid_all_vars
		prev_all_vars = parent.prev_all_vars

		if self.N != self.Ntilde:
			self.obj_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.tf], Obj, order='F')

			# Gradient vector ("jac")
			obj_jac = Matrix(fast_jac([Obj], prev_all_vars+all_vars + mid_all_vars+[self.tf])).T
			self.obj_jac_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.tf], obj_jac, order='F')

			# hessian matrix ("hess")
			obj_hess = Matrix(fast_half_hess(Obj, prev_all_vars+all_vars + mid_all_vars+[self.tf])).T
			self.obj_hess_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.tf], obj_hess, order='F')

		else:
			self.obj_lambda = Lambdify(all_vars+[self.tf], Obj, order='F')

			# Gradient vector ("jac")
			obj_jac = Matrix([Obj]).jacobian(all_vars+[self.tf])
			self.obj_jac_lambda = Lambdify(all_vars+[self.tf], obj_jac, order='F')

			# hessian matrix ("hess")
			obj_hess = hessian(Obj, all_vars+[self.tf])
			self.obj_hess_lambda = Lambdify(all_vars+[self.tf], obj_hess, order='F')

		x0 = np.ones(self.Ntilde * (self.X_dim + self.U_dim) + 1)
		self.hess_shape = (x0.size, x0.size)
		self.hess_sparse_indices = self.hess(x0, return_sparse_indices=True)        		
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
		arg_x = arg[:-1]
		arg_tf = arg[-1]
		if self.N != self.Ntilde:
			_tf = arg_tf * np.ones((self.N-1,1))
			V = arg_x[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg_x[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:], _tf))
		else:
			_tf = arg_tf * np.ones((self.N, 1))
			_in = np.hstack((arg_x.reshape(self.Ntilde, self.X_dim+self.U_dim), _tf)) 

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

		arg_x = arg[:-1]
		arg_tf = arg[-1]
		if self.N != self.Ntilde:
			_tf = arg_tf * np.ones((self.N-1, 1))
			V = arg_x[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg_x[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:], _tf))
			J = self.obj_jac_lambda(_in.T).squeeze()
			SysDim = self.X_dim + self.U_dim
			jac = np.zeros(self.Ntilde * SysDim + 1)
			for i in range(self.N-1):
				jac[i*SysDim:(i+1)*SysDim+SysDim] += J[:SysDim*2,i]
				jac[(i+self.N)*SysDim:(i+self.N)*SysDim+SysDim] += J[SysDim*2:-1,i]
				jac[-1] += J[-1,i]
		else:
			_tf = arg_tf * np.ones((self.N, 1))
			_in = np.hstack((arg_x.reshape(self.Ntilde, self.X_dim+self.U_dim), _tf)) 
			J = self.obj_jac_lambda(_in.T).squeeze()
			SysDim = self.X_dim + self.U_dim
			jac = np.zeros(self.N * SysDim + 1)
			for i in range(self.N):
				jac[i*SysDim:i*SysDim+SysDim] += J[:SysDim,i]
				jac[-1] += J[-1,i]

		return jac

	def hess(self, arg, return_sparse_indices=False):
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

		arg_x = arg[:-1]
		arg_tf = arg[-1]

		Sys_dim = self.X_dim + self.U_dim
		Opt_dim = Sys_dim * self.Ntilde
		if self.N != self.Ntilde:
			_tf = arg_tf * np.ones((self.N-1, 1))
			V = arg_x[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
			Vmid = arg_x[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
			_in = np.hstack((V[:-1,:], Vmid, V[1:,:], _tf))

			hess_block = self.obj_hess_lambda(_in.T) + 1e-9

			# used for determining nonzero elements of hessian
			if return_sparse_indices:
				idx = set()
				for i in range(self.N-1):
					for j in range(3*Sys_dim+1):
						for k in range(j, 3*Sys_dim+1):
							if j < 2*Sys_dim and k < 2*Sys_dim: # A
								idx.add((i*Sys_dim+j, i*Sys_dim+k))
								idx.add((i*Sys_dim+k, i*Sys_dim+j))								
							elif j >= 2*Sys_dim and j < 3*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # B
								idx.add(((i + self.N)*Sys_dim+j-2*Sys_dim, (i + self.N)*Sys_dim+k-2*Sys_dim))
								idx.add(((i + self.N)*Sys_dim+k-2*Sys_dim, (i + self.N)*Sys_dim+j-2*Sys_dim))
							elif j < 2*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # C == D
								idx.add((i*Sys_dim+j, (i + self.N)*Sys_dim+k-2*Sys_dim))
								idx.add(((i + self.N)*Sys_dim+k-2*Sys_dim, i*Sys_dim+j))
							elif j < 2*Sys_dim and k == 3*Sys_dim: # E==F
								idx.add((i*Sys_dim+j, arg.size-1))
								idx.add((arg.size-1, i*Sys_dim+j))
							elif j >= 2*Sys_dim and k == 3*Sys_dim: # E==F
								idx.add(((i + self.N)*Sys_dim+j-2*Sys_dim, arg.size-1))
								idx.add((arg.size-1, (i + self.N)*Sys_dim+j-2*Sys_dim))
							else:
								idx.add((arg.size-1,arg.size-1))
				idx = np.array(list(idx))
				return idx[:,0], idx[:,1]
			else:
				hess = np.zeros(self.hess_size, dtype=np.float)
				for i in range(self.N-1):
					Htemp = hess_block[:,:,i] + hess_block[:,:,i].T
					for j in range(3*Sys_dim+1):
						for k in range(j, 3*Sys_dim+1):
							if j < 2*Sys_dim and k < 2*Sys_dim: # A
								hess[self.hess_dict[(i*Sys_dim+j, i*Sys_dim+k)]]+=Htemp[j,k]
								hess[self.hess_dict[(i*Sys_dim+k, i*Sys_dim+j)]]+=Htemp[k,j]
							elif j >= 2*Sys_dim and j < 3*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # B
								hess[self.hess_dict[((i + self.N)*Sys_dim+j-2*Sys_dim, (i + self.N)*Sys_dim+k-2*Sys_dim)]]+=Htemp[j,k]
								hess[self.hess_dict[((i + self.N)*Sys_dim+k-2*Sys_dim, (i + self.N)*Sys_dim+j-2*Sys_dim)]]+=Htemp[k,j]
							elif j < 2*Sys_dim and k >= 2*Sys_dim and k < 3*Sys_dim: # C == D
								hess[self.hess_dict[(i*Sys_dim+j, (i + self.N)*Sys_dim+k-2*Sys_dim)]]+=Htemp[j,k]
								hess[self.hess_dict[((i + self.N)*Sys_dim+k-2*Sys_dim, i*Sys_dim+j)]]+=Htemp[k,j]
							elif j < 2*Sys_dim and k == 3*Sys_dim: # E==F
								hess[self.hess_dict[(i*Sys_dim+j, arg.size-1)]]+=Htemp[j,k]
								hess[self.hess_dict[(arg.size-1, i*Sys_dim+j)]]+=Htemp[k,j]
							elif j >= 2*Sys_dim and k == 3*Sys_dim: # E==F
								hess[self.hess_dict[((i + self.N)*Sys_dim+j-2*Sys_dim, arg.size-1)]]+=Htemp[j,k]
								hess[self.hess_dict[(arg.size-1, (i + self.N)*Sys_dim+j-2*Sys_dim)]]+=Htemp[k,j]
							else:
								hess[self.hess_dict[(arg.size-1,arg.size-1)]]+=Htemp[j,k]
				return coo_matrix((hess, self.hess_sparse_indices),shape=self.hess_shape)
		else:
			_tf = arg_tf * np.ones((self.N, 1))
			_in = np.hstack((arg_x.reshape(self.Ntilde, self.X_dim+self.U_dim), _tf)) 
			hess_block = self.obj_hess_lambda(_in.T) + 1e-9
			# used for determining nonzero elements of hessian
			if return_sparse_indices:
				idx = set()
				for i in range(self.N):
					for j in range(Sys_dim+1):
						for k in range(j, Sys_dim+1):
							if j < Sys_dim and k < Sys_dim: # A								
								idx.add((i*Sys_dim+j, i*Sys_dim+k))
								idx.add((i*Sys_dim+k, i*Sys_dim+j))
							elif j < Sys_dim and k == Sys_dim: # E==F
								idx.add((i*Sys_dim+j, arg.size-1))
								idx.add((arg.size-1, i*Sys_dim+j))
							else:
								idx.add((arg.size-1,arg.size-1))

				idx = np.array(list(idx))
				return idx[:,0], idx[:,1]
			else:
				hess = np.zeros(self.hess_size, dtype=np.float)

				for i in range(self.N):
					Htemp = hess_block[:,:,i] + hess_block[:,:,i].T
					for j in range(Sys_dim+1):
						for k in range(j, Sys_dim+1):
							if j < Sys_dim and k < Sys_dim: # A
								hess[self.hess_dict[(i*Sys_dim+j, i*Sys_dim+k)]]=Htemp[j,k]
								hess[self.hess_dict[(i*Sys_dim+k, i*Sys_dim+j)]]=Htemp[k,j]
							elif j < Sys_dim and k == Sys_dim: # E==F
								hess[self.hess_dict[(i*Sys_dim+j, arg.size-1)]]=Htemp[j,k]
								hess[self.hess_dict[(arg.size-1, i*Sys_dim+j)]]=Htemp[k,j]
							else:
								hess[self.hess_dict[(arg.size-1,arg.size-1)]]+=Htemp[j,k]
				return coo_matrix((hess, self.hess_sparse_indices),shape=self.hess_shape)
