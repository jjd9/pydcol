"""

Fast symbolic operations implemented using symengine.

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

# third party imports
from symengine import zeros, diff

def fast_jac(expr, vs):
    """
    Evaluate the jacobian of expr w.r.t. the variables in vs
    """
    J = zeros(len(expr), len(vs))
    for i in range(len(expr)):
        for j in range(len(vs)):
            J[i, j] = diff(expr[i], vs[j])
    return J

def fast_hess(expr, vs):
    """
    Evaluate the hessian of expr w.r.t. the variables in vs
    """
    H = zeros(len(vs), len(vs))
    for i in range(len(vs)):
        for j in range(len(vs)):
            if i > j:
                H[i, j] = H[j, i]
            else:
                H[i, j] = diff(expr, vs[i], vs[j])
    return H

def fast_half_hess(expr, vs):
    """
    Evaluate the upper triangular portion of the hessian of expr 
    w.r.t. the variables in vs. Diagonal elements are divided by 2 such that:
    H_half + H_halt.T = H
    """
    H = zeros(len(vs), len(vs))
    for i in range(len(vs)):
        for j in range(i, len(vs)):
            H[i, j] = diff(expr, vs[i], vs[j])
            if i == j:
                H[i, j] *= 0.5
    return H
