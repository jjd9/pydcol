# third party imports
from pydcol.CollocMethods import MIDPOINT_METHODS
import numpy as np
from symengine import Lambdify
from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise

from .SymUtils import fast_jac, fast_half_hess
from scipy.sparse import csr_matrix, lil_matrix

class Objective:
    def __init__(self, parent, Obj):
        self.N = parent.N
        self.Ntilde = parent.Ntilde
        self.h = parent.h
        self._h = parent._h

        self.colloc_method = parent.colloc_method

        self.X_dim = parent.X_dim
        self.U_dim = parent.U_dim

        all_vars = parent.all_vars
        mid_all_vars = parent.mid_all_vars
        prev_all_vars = parent.prev_all_vars

        if self.colloc_method in MIDPOINT_METHODS:
            self.obj_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], Obj, order='F')

            # Gradient vector ("jac")
            obj_jac = Matrix(fast_jac([Obj], prev_all_vars+all_vars + mid_all_vars)).T
            self.obj_jac_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], obj_jac, order='F')

            # hessian matrix ("hess")
            obj_hess = Matrix(fast_half_hess(Obj, prev_all_vars+all_vars + mid_all_vars)).T
            self.obj_hess_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], obj_hess, order='F')
        else:
            self.obj_lambda = Lambdify(all_vars, Obj, order='F')

            # Gradient vector ("jac")
            obj_jac = Matrix([Obj]).jacobian(all_vars)
            self.obj_jac_lambda = Lambdify(all_vars, obj_jac, order='F')

            # hessian matrix ("hess")
            obj_hess = hessian(Obj, all_vars)
            self.obj_hess_lambda = Lambdify(all_vars, obj_hess, order='F')

        x0 = np.ones(self.Ntilde * (self.X_dim + self.U_dim))
        self.hess_sparse_indices = self.hess(x0, fill=True)        

    # create callback for scipy
    def eval(self, arg):
        if self.colloc_method in MIDPOINT_METHODS:
            V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
            Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
            _in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))
        else:
            _in = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)

        return self.obj_lambda(_in.T).sum()

    def jac(self, arg):
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
            _in = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)
            jac = self.obj_jac_lambda(_in.T).squeeze().T.ravel()

        return jac

    def hess(self, arg, fill=False):
        Sys_dim = self.X_dim + self.U_dim
        Opt_dim = Sys_dim * self.Ntilde
        if self.N != self.Ntilde:
            V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
            Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
            _in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))

            hess_block = self.obj_hess_lambda(_in.T)

            # used for determining nonzero elements of hessian
            if fill:
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
                hess = lil_matrix((arg.size, arg.size), dtype=np.float)
                for i in range(self.N-1):
                    Htemp = hess_block[:,:,i] + hess_block[:,:,i].T
                    for j in range(2*Sys_dim):
                        for k in range(2*Sys_dim):
                            hess[i*Sys_dim+j, i*Sys_dim+k]+=Htemp[j,k]
                    for j in range(Sys_dim):
                        for k in range(Sys_dim):
                            hess[(i + self.N)*Sys_dim+j, (i + self.N)*Sys_dim+k]+=Htemp[2*Sys_dim+j,2*Sys_dim+k]
                return hess
        else:
            V = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)
            hess_block = self.obj_hess_lambda(V.T)
            # used for determining nonzero elements of hessian
            if fill:
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