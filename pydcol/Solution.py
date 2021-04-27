"""

Class for storing collocation solutions

"""

# third party imports
from scipy.interpolate import interp1d

# pydcol imports
from .CollocMethods import *

class Solution:

	def __init__(self, sol, colloc_method, dims, tspan, solver):
		(N, Ntilde, X_dim, U_dim) = dims
		self.t = tspan

		if solver == 'ipopt':
			# save whether or not the optimization succeeded
			self.success = True
			V = sol.reshape(Ntilde, X_dim+U_dim)
		else:
			# save whether or not the optimization succeeded
			self.success = sol.success
			V = sol.x.reshape(Ntilde, X_dim+U_dim)

		self.x = V[:N, :X_dim]
		self.u = V[:N, X_dim:X_dim+U_dim]

		# convert discrete control to time-varying spline
		if colloc_method == TRAP:
			self.u_t = interp1d(tspan, self.u.ravel(), kind='linear') # linear for trapezoid method
		elif colloc_method == HERM:			
			self.u_t = interp1d(tspan, self.u.ravel(), kind='quadratic') # quadratic for hermite simpson method
		elif colloc_method == RADAU:			
			self.u_t = interp1d(tspan, self.u.ravel(), kind='cubic') # quadratic for hermite simpson method