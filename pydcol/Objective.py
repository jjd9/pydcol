# third party imports
from pydcol.CollocMethods import MIDPOINT_METHODS
import numpy as np
from symengine import Lambdify
from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise

from .SymUtils import fast_jac, fast_half_hess

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

    # create callback for scipy
    def eval(self, arg):
        # this works fine because order doesn't matter for our objective
        if self.colloc_method in MIDPOINT_METHODS:
            V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
            Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
            _in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))
        else:
            _in = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)

        return self.obj_lambda(_in.T).sum()

    def jac(self, arg):
        # this works fine because order doesn't matter for our objective
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
        # this works fine because order doesn't matter for our objective
        Sys_dim = self.X_dim + self.U_dim
        Opt_dim = Sys_dim * self.Ntilde
        hess = np.zeros((Opt_dim, Opt_dim), dtype=np.float)
        if self.N != self.Ntilde:
            V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
            Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
            _in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))

            hess_block = self.obj_hess_lambda(_in.T)

            # used for determining nonzero elements of hessian
            if fill:
                hess_block[:,:,:] = 1.0

            for i in range(self.N-1):
                Htemp = hess_block[:,:,i] + hess_block[:,:,i].T
                hess[i*Sys_dim:(i+1)*Sys_dim + Sys_dim, i*Sys_dim:(i+1)*Sys_dim + Sys_dim] += Htemp[:Sys_dim*2,:Sys_dim*2]
                hess[(i + self.N)*Sys_dim:(i + self.N)*Sys_dim + Sys_dim, (i + self.N)*Sys_dim:(i + self.N)*Sys_dim + Sys_dim] += Htemp[Sys_dim*2:,Sys_dim*2:]
        else:
            V = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)
            hess_block = self.obj_hess_lambda(V.T)
            for i in range(self.N-1):
                hess[i*Sys_dim:i*Sys_dim + Sys_dim, i*Sys_dim:i*Sys_dim + Sys_dim] += hess_block[:Sys_dim,:Sys_dim,i]

        return hess
