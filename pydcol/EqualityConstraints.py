# third party imports
from re import X
import numpy as np
from symengine import Lambdify
from sympy import Matrix, hessian, Symbol, symbols, lambdify
from sympy.matrices.dense import matrix_multiply_elementwise

# pydcol imports
from .SymUtils import fast_jac, fast_half_hess

from scipy.sparse import csr_matrix, lil_matrix

class EqualityConstraints:
    def __init__(self, parent, C_eq):
        self.N = parent.N
        self.Ntilde = parent.Ntilde
        
        self.colloc_method = parent.colloc_method

        self.X_dim = parent.X_dim
        self.U_dim = parent.U_dim

        self.h = parent.h
        self._h = parent._h

        self.X_start = parent.X_start
        self.X_goal = parent.X_goal

        all_vars = parent.all_vars
        prev_all_vars = parent.prev_all_vars
        mid_all_vars = parent.mid_all_vars

        self.ceq_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], C_eq, order='F')

        # jacobian matrix ("jac")
        print("Jacobian")
        ceq_jac = Matrix(fast_jac(C_eq, prev_all_vars+all_vars + mid_all_vars)).T
        print("Jacobian lambdify")
        self.ceq_jac_lambda = Lambdify(prev_all_vars+mid_all_vars+all_vars+[self.h], ceq_jac, order='F')

        # Hessian Matrix ("hess")
        print("Hessian")
        # lagrange multipliers
        self.ncon = len(C_eq)
        lamb = Matrix(["lambda" + str(i) for i in range(self.ncon)]).reshape(self.ncon, 1)
        ceq_hess = Matrix(fast_half_hess((C_eq.T * lamb)[0], prev_all_vars + all_vars + mid_all_vars))
        print("Hessian lambdify")
        self.ceq_hess_lamb = Lambdify(prev_all_vars+mid_all_vars+all_vars+list(lamb)+[self.h], ceq_hess, order='F', cse=True, backend='llvm')

        # linear if hessian is all zero
        if len(ceq_hess.free_symbols) == 0:
            self.is_linear = True
        else:
            self.is_linear = False

        x0 = np.ones(self.Ntilde * (self.X_dim + self.U_dim))
        self.jac_sparse_indices = self.jac(x0, fill=True)

        ncon = self.eval(x0).size
        lagrange = np.ones(ncon)
        self.hess_sparse_indices = self.hess(x0, lagrange, fill=True)
        
    def eval(self, arg):
        if self.N == self.Ntilde:
            V = arg.reshape(self.N, self.X_dim+self.U_dim)
            _X = V[:,:self.X_dim]
            _in = np.hstack((V[:-1,:], V[1:,:],self._h.reshape(-1,1)))
        else:
            V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
            Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
            _X = V[:,:self.X_dim]
            _in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))
        _out = self.ceq_lambda(_in.T).T.ravel()
        initial_constr = (_X[0,:] - self.X_start).ravel()
        terminal_constr = (_X[-1,:] - self.X_goal).ravel()
        return np.hstack((_out, initial_constr, terminal_constr))

    def jac(self, arg, fill=False):
        if self.N == self.Ntilde:
            V = arg.reshape(self.Ntilde, self.X_dim+self.U_dim)
            _in = np.hstack((V[:-1,:], V[1:,:],self._h.reshape(-1,1)))
        else:
            V = arg[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
            Vmid = arg[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
            _in = np.hstack((V[:-1,:], Vmid, V[1:,:],self._h.reshape(-1,1)))
        J = self.ceq_jac_lambda(_in.T)

        # jac should be Num_constraints x Opt_dim
        Opt_dim = (self.X_dim + self.U_dim)
        Ceq_dim = self.ncon
        jac_shape = (Ceq_dim * (self.N-1) + 2 * self.X_dim, Opt_dim * self.Ntilde)

        # used for determining nonzero elements of jacobian
        if fill:
            rows = []
            cols = []
            for i in range(self.N-1):
                for j in range(i*Ceq_dim, i*Ceq_dim + Ceq_dim):
                    for k in range(i*Opt_dim, (i+1)*Opt_dim + Opt_dim):
                        rows.append(j)
                        cols.append(k)
                if self.N != self.Ntilde:
                    for j in range(i*Ceq_dim, i*Ceq_dim + Ceq_dim):
                        for k in range((i + self.N)*Opt_dim, (i + self.N)*Opt_dim + Opt_dim):
                            rows.append(j)
                            cols.append(k)
            # initial and terminal constraint gradients are easy
            rows += np.arange(Ceq_dim * (self.N-1), Ceq_dim * (self.N-1) + self.X_dim).tolist()
            rows += np.arange(Ceq_dim * (self.N-1) + self.X_dim, jac_shape[0]).tolist()
            cols += np.arange(0, self.X_dim).tolist()
            cols += np.arange(jac_shape[1]-(self.N-1)*(self.X_dim+self.U_dim)-(self.X_dim+self.U_dim), jac_shape[1]-(self.N-1)*(self.X_dim+self.U_dim)-self.U_dim).tolist()
            return rows, cols
        else:
            jac = []
            for i in range(self.N-1):
                jac += J[:2*Opt_dim,:,i].T.ravel().tolist()
                if self.N != self.Ntilde:
                    jac += J[2*Opt_dim:,:,i].T.ravel().tolist()
            # initial and terminal constraint gradients are easy
            jac += np.ones(2 * self.X_dim).tolist()
            jac = csr_matrix((jac,self.jac_sparse_indices),shape=jac_shape)
            return jac

    def hess(self, arg_x, arg_v, fill=False):
        hess_shape = (arg_x.size, arg_x.size)
        if self.is_linear:
            if fill:
                return [], []
            else:
                return csr_matrix(hess_shape)
        else:
            if self.N == self.Ntilde:
                V = arg_x.reshape(self.N, self.X_dim+self.U_dim)
                _L = arg_v[:-2*self.X_dim].reshape(self.N-1, self.ncon)
                _in = np.hstack((V[:-1,:], V[1:,:], _L, self._h.reshape(-1,1)))
            else:
                V = arg_x[:self.N * (self.X_dim+self.U_dim)].reshape(self.N, self.X_dim+self.U_dim)
                Vmid = arg_x[self.N * (self.X_dim+self.U_dim):].reshape(self.N - 1, self.X_dim+self.U_dim)
                _L = arg_v[:-2*self.X_dim].reshape(self.N-1, self.ncon)
                _in = np.hstack((V[:-1,:], Vmid, V[1:,:], _L, self._h.reshape(-1,1)))

            H = self.ceq_hess_lamb(_in.T)

            # used for determining nonzero elements of hessian
            Opt_dim = (self.X_dim + self.U_dim)

            if fill:
                idx = set()
                for i in range(self.N-1):
                    for j in range(i*Opt_dim, (i+1)*Opt_dim + Opt_dim):
                        for k in range(i*Opt_dim, (i+1)*Opt_dim + Opt_dim):
                            idx.add((j, k))
                    if self.N != self.Ntilde:
                        for j in range(Opt_dim):
                            for k in range(Opt_dim):
                                idx.add(((i + self.N)*Opt_dim+j, (i + self.N)*Opt_dim+k))
                idx = np.array(list(idx))
                rows = idx[:,0]
                cols = idx[:,1]
                return rows, cols
            else:
                hess = lil_matrix((arg_x.size, arg_x.size), dtype=np.float)
                for i in range(self.N-1):
                    Htemp = H[:,:,i] + H[:,:,i].T
                    for j in range(2*Opt_dim):
                        for k in range(2*Opt_dim):
                            hess[i*Opt_dim+j, i*Opt_dim+k]+=Htemp[j,k]
                    for j in range(Opt_dim):
                        for k in range(Opt_dim):
                            hess[(i + self.N)*Opt_dim+j, (i + self.N)*Opt_dim+k]+=Htemp[2*Opt_dim+j,2*Opt_dim+k]
                return hess
