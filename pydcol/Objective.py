# third party imports
import numpy as np
from symengine import Lambdify
from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise

class Objective:
    def __init__(self, parent, Obj):
        self.N = parent.N
        self.Ntilde = parent.Ntilde

        self.X_dim = parent.X_dim
        self.U_dim = parent.U_dim

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
        # this works fine because order doesn't matter for our objective
        V = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)
        return self.obj_lambda(V.T).sum()

    def jac(self, arg):
        # this works fine because order doesn't matter for our objective
        V = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)
        return self.obj_jac_lambda(V.T).squeeze().T.ravel()

    def hess(self, arg, fill=False):
        # this works fine because order doesn't matter for our objective
        Sys_dim = self.X_dim + self.U_dim
        Opt_dim = Sys_dim * self.Ntilde
        hess = np.zeros((Opt_dim, Opt_dim), dtype=np.float)
        V = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)
        hess_block = self.obj_hess_lambda(V.T)

        # used for determining nonzero elements of hessian
        if fill:
            hess_block[:,:,:] = 1.0

        for i in range(self.Ntilde):
            hess[i*Sys_dim:i*Sys_dim + Sys_dim, i*Sys_dim:i*Sys_dim + Sys_dim] = hess_block[:,:,i]
        return hess
