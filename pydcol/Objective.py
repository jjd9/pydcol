# third party imports
import numpy as np
from symengine import Lambdify
from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise

class Objective:
    def __init__(self, parent, Obj):
        self.N = parent.N
        self.X_dim = parent.X_dim
        self.U_dim = parent.U_dim

        X_sym = parent.X_sym
        U_sym = parent.U_sym
        all_vars = parent.all_vars

        self.obj_lambda = Lambdify(all_vars, Obj, order='F')

        # Gradient vector ("jac")
        obj_jac = Matrix([Obj]).jacobian(all_vars)
        self.obj_jac_lambda = Lambdify(all_vars, obj_jac, order='F')

        # hessian matrix ("hess")
        obj_hess = hessian(Obj, all_vars)
        self.obj_hess_lambda = Lambdify(all_vars, obj_hess, order='F')

    # create callback for scipy
    def eval(self, arg):
        V = arg.reshape(self.N, self.X_dim+self.U_dim)
        return self.obj_lambda(V.T).sum()

    def jac(self, arg):
        V = arg.reshape(self.N, self.X_dim+self.U_dim)
        return self.obj_jac_lambda(V.T).squeeze().T.ravel()

    def hess(self, arg):
        V = arg.reshape(self.N, self.X_dim+self.U_dim)
        hess_block = self.obj_hess_lambda(V.T)
        Sys_dim = self.X_dim + self.U_dim
        Opt_dim = Sys_dim * self.N
        hess = np.zeros((Opt_dim, Opt_dim))
        for i in range(self.N):
            hess[i*Sys_dim:i*Sys_dim + Sys_dim, i*Sys_dim:i*Sys_dim + Sys_dim] = hess_block[:,:,i]
        return hess
