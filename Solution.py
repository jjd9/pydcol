"""

Class for storing collocation solutions

"""

from scipy.interpolate import interp1d

from CollocMethods import *

class Solution:

	def __init__(self, scipy_sol, colloc_method, dims, tspan):
		(N, X_dim, U_dim) = dims
		# save whether or not the optimization succeeded
		self.success = scipy_sol.success
		self.t = tspan
		
		# extract solution at collocation points
		V = scipy_sol.x.reshape(N, X_dim+U_dim)
		self.x = V[:, :X_dim]
		self.u = V[:, X_dim:X_dim+U_dim]

		# convert discrete control to time-varying spline
		if colloc_method == TRAP:
			self.u_t = interp1d(tspan, self.u.ravel(), kind='linear') # linear for trapezoid method
		elif colloc_method == HERM:
			self.u_t = interp1d(tspan, self.u.ravel(), kind='quadratic') # quadratic for hermite simpson method