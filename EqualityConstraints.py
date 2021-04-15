from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise

from symengine import zeros, diff, sympify
from symengine import Lambdify

import numpy as np

def fast_jac(expr, vs):
    J = zeros(len(expr), len(vs))
    for i in range(len(expr)):
        for j in range(len(vs)):
            J[i, j] = diff(expr[i], vs[j])
    return J

def fast_hess(expr, vs):
    H = zeros(len(vs), len(vs))
    for i in range(len(vs)):
        for j in range(len(vs)):
            if i > j:
                H[i, j] = H[j, i]
            else:
                H[i, j] = diff(expr, vs[i], vs[j])
    return H

def fast_half_hess(expr, vs):
    H = zeros(len(vs), len(vs))
    for i in range(len(vs)):
        for j in range(i, len(vs)):
            H[i, j] = diff(expr, vs[i], vs[j])
            if i == j:
                H[i, j] *= 0.5
    return H


class EqualityConstraints:
    def __init__(self, parent, C_eq):
        self.N = parent.N
        self.X_dim = parent.X_dim
        self.U_dim = parent.U_dim

        self.h = parent.h
        self._h = parent._h

        self.X_start = parent.X_start
        self.X_goal = parent.X_goal

        all_vars = parent.all_vars
        prev_all_vars = parent.prev_all_vars

        self.ceq_lambda = Lambdify(prev_all_vars+all_vars+[self.h], C_eq, order='F')

        # jacobian matrix ("jac")
        print("Jacobian")
        ceq_jac = Matrix(fast_jac(C_eq, prev_all_vars + all_vars)).T
        print("Jacobian lambdify")
        self.ceq_jac_lambda = Lambdify(prev_all_vars+all_vars+[self.h], ceq_jac, order='F')

        # Hessian Matrix ("hess")
        print("Hessian")
        # lagrange multipliers
        lamb = Matrix(["lambda" + str(i) for i in range(self.X_dim)]).reshape(self.X_dim, 1)
        ceq_hess = Matrix(fast_half_hess((C_eq.T * lamb)[0], prev_all_vars + all_vars))
        print("Hessian lambdify")
        # self.ceq_hess_lamb = Lambdify(prev_all_vars+all_vars+list(lamb)+[self.h], ceq_hess, order='F')
        self.ceq_hess_lamb = Lambdify(prev_all_vars+all_vars+list(lamb)+[self.h], ceq_hess, order='F', cse=True, backend='llvm')

        # linear if hessian is all zero
        if len(ceq_hess.free_symbols) == 0:
            self.is_linear = True
        else:
            self.is_linear = False

    def eval(self, arg):
        V = arg.reshape(self.N, self.X_dim+self.U_dim)
        _X = V[:,:self.X_dim]
        _in = np.hstack((V[:-1,:], V[1:,:],self._h.reshape(-1,1)))
        _out = self.ceq_lambda(_in.T).T.ravel()
        initial_constr = (_X[0,:] - self.X_start).ravel()
        terminal_constr = (_X[-1,:] - self.X_goal).ravel()
        return np.hstack((_out, initial_constr, terminal_constr))

    def jac(self, arg):
        V = arg.reshape(self.N, self.X_dim+self.U_dim)
        _in = np.hstack((V[:-1,:], V[1:,:],self._h.reshape(-1,1)))
        J = self.ceq_jac_lambda(_in.T)

        # jac should be Num_constraints x Opt_dim
        Opt_dim = (self.X_dim + self.U_dim)
        Ceq_dim = self.X_dim
        jac = np.zeros((Ceq_dim * (self.N-1) + 2 * self.X_dim, Opt_dim * self.N))
        for i in range(self.N-1):
            jac[i*Ceq_dim:i*Ceq_dim + Ceq_dim, i*Opt_dim:(i+1)*Opt_dim + Opt_dim] = J[:,:,i].T
        # initial and terminal constraint gradients are easy
        jac[Ceq_dim * (self.N-1):Ceq_dim * (self.N-1) + self.X_dim, :self.X_dim] = np.eye(self.X_dim)
        jac[Ceq_dim * (self.N-1) + self.X_dim:,-(self.X_dim+self.U_dim):-self.U_dim] = np.eye(self.X_dim)
        return jac

    def hess(self, arg_x, arg_v):
        if self.is_linear:
            hess = np.zeros((arg_x.size, arg_x.size), dtype=np.float)
        else:
            V = arg_x.reshape(self.N, self.X_dim+self.U_dim)
            _L = arg_v[:-2*self.X_dim].reshape(self.N-1, self.X_dim)
            _in = np.hstack((V[:-1,:], V[1:,:], _L, self._h.reshape(-1,1)))
            H = self.ceq_hess_lamb(_in.T)

            # Reshape the lagrange multiplier vector
            hess = np.zeros((arg_x.size, arg_x.size), dtype=np.float)

            Opt_dim = (self.X_dim + self.U_dim)

            for i in range(self.N-1):
                hess[i*Opt_dim:(i+1)*Opt_dim + Opt_dim, i*Opt_dim:(i+1)*Opt_dim + Opt_dim] += H[:,:,i] + H[:,:,i].T

        return hess
