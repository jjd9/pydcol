"""

Class for storing collocation solutions

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

# third party imports
import numpy as np
from scipy.interpolate import interp1d

# pydcol imports
from .CollocMethods import *

class Solution:

	def __init__(self, sol, colloc_method, dims, tspan, solver):
		(N, Ntilde, X_dim, U_dim) = dims

		# save whether or not the optimization succeeded
		if solver == 'ipopt':
			self.opt_x = sol
			self.success = True
			V = sol.reshape(Ntilde, X_dim+U_dim)
		else:
			self.opt_x = sol.x
			self.success = sol.success
			V = sol.x.reshape(Ntilde, X_dim+U_dim)

		self.x_endpts = V[:N, :X_dim].copy()
		self.u_endpts = V[:N, X_dim:].copy()

		if N != Ntilde:
			# put points in the right order
			Vtemp = [V[0,:]]
			tspan_temp = [tspan[0]]
			for i in range(1,N):
				Vtemp += [V[N+i-1,:]]
				Vtemp += [V[i,:]]
				tspan_temp+=[tspan[i-1] + 0.5*(tspan[i]-tspan[i-1])]
				tspan_temp+=[tspan[i]]
			V = np.array(Vtemp)
			tspan = np.array(tspan_temp)

		self.t = tspan
		self.x = V[:, :X_dim]
		self.u = V[:, X_dim:]

		# convert discrete control to time-varying spline
		if colloc_method in [TRAP, EB, EF]:
			self.u_t = interp1d(tspan, self.u.T, kind='linear') # linear for trapezoid method
		elif colloc_method == HERM:			
			self.u_t = interp1d(tspan, self.u.T, kind='quadratic') # quadratic for hermite simpson method
		elif colloc_method == RADAU:			
			self.u_t = interp1d(tspan, self.u.T, kind='cubic') # cubic for 3rd order radau method