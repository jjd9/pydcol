# pydcol
This is a repository for our ME 397 Numerical methods project.

pydcol solves optimal control problems using direct collocation. The specific problem pydcol solves for a given system is to find an open-loop control trajectory connecting an initial system state to a final system state over a fixed time interval. The control trajectory provided by pydcol minimizes the control effort, which is the integral of the sum of all the control inputs squared.

<!-- $ \int_{t_{0}}^{t_{f}} u^2 \,dt $ --> <img style="transform: translateY(0.1em); background: white;" src="svg/u1sApyE8bZ.svg">

This is accomplished by converting the continuous ode system into a finite dimensional nonlinear optimization problem (NLP) using an integration scheme. This process is called direct collocation or simultaneous discretization.

## Getting Started
Please make sure you have Python 3 installed on your computer. pydcol was developed in Python 3.8, but earlier versions of Python 3 should work.

It is always recommended to setup a virtual environment so that our dependencies do not mess up your system setup. To do so, run:

On Mac/Linux systems:
```
python3 -m venv pydcol-env
source pydcol-env/bin/activate
```

On Windows:
```
python -m venv pydcol-env
pydcol-env\Scripts\activate.bat
```

The python modules can then be installed using: 
```pip install -r requirements.txt```

If you want to use IPOPT, extra steps are involved for installing ipyopt on non-linux systems. Please see the package for more information: https://gitlab.com/g-braeunlich/ipyopt. However, this is not mandatory to use pydcol.

To confirm the setup worked properly, run the BlockMove.py example as follows:
```
python BlockMove.py
```

## Usage
1.) Define your ode model as a list of sympy expressions. The list should only contain the right-hand-side of the ode equations. So if your system was:
```
dx/dt = y + u
dy/dt = x^2 + y
```
your list should be: ode = [y + u, x^2 + y]

2.) Distinguish control variables from state variables. This is so that the solver can properly compute the effort objective. For this toy example, u is the only control variable so:
```
state_vars = [x, y]
control_vars = [u]
```
3.) Define a start and goal state for the system. These should be numpy arrays each with the same dimensions as state_vars.
```
X_start = np.array([0, 0]) # arbitrary goal state
X_goal = np.array([1, 0]) # arbitrary goal state
```
4.) Set bounds on the variables. Use None if the variable is not bounded.
```
# bounds = [[lb_x, ub_x],[lb_y, ub_y],[lb_u, ub_u]]
u_max = 10
bounds = [[None,None],[None,None],[-u_max, u_max]]
```

5.) Decide how many nodes to use. More nodes means better accuracy but possibly a longer solve time. pydcol accepts arbitrary node spacings, but does expect that tspan is sorted from t0 to tf and has no repeats.
```
t0= 0
tf = 1
N = 10
tspan = np.linspace(0,1,N)
```
6.) Define the problem. At this stage, your problem is converted into an objective and constraint function (along with functions to evalute the 1st and second derivatives of those functions). Multiple collocation methods are supported. Here we use euler-forward. For a full list of methods, please see pydol/CollocationMethods.py.
```
problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tspan, colloc_method=EF)
```
7.) Solve the problem. The scipy.minimize,trust-constr (https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html) and IPOPT (https://github.com/coin-or/Ipopt) solvers are supported. IPOPT takes faster steps and handles large, sparse systems better (>6 states and/or 1000's of nodes). scipy.minimize.trust-constr takes more intelligent steps and handles small systems better (<6 states and/or 100's of nodes). 
```
sol_c = problem.solve(bounds=bounds, solver='scipy')
```
It is up to the user to ask reasonable things of the optimizer. Ensure that it is physically possible for your system to go from the state X_start to X_goal in the provided time bounds. A good initial guess is not always mandatory but often helps.

8.) Compare the solution to an IVP solution. The control trajectory from the collocation solution found in step 7 is used to integrate the system from t0 to tf. scipy.integrate.solve_ivp is used. The ivp solver can be selected using ivp_method. We recommend implicit methods unless the control trajectory is very smooth.
```
problem.evaluate(ivp_method='Radau')
```

Please see the examples for more illustrations of how to use the library.

## Third-party Software Acknowledgements
Matplotlib (https://matplotlib.org/)
- Plots and animations

Numpy (https://numpy.org/)
- np.array datastructure
- Fast operations on np.arrays

Ipyopt (https://gitlab.com/g-braeunlich/ipyopt)
- Wrapper around IPOPT for Nonlinear optimization (https://github.com/coin-or/Ipopt)

Scipy (for supporting numerical methods):
- Sparse matrix manipulation (https://docs.scipy.org/doc/scipy/reference/sparse.html)
- Solving IVP's (https://docs.scipy.org/doc/scipy/reference/integrate.html)
- Nonlinear optimization (https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html)

Symengine (https://github.com/symengine/symengine.py):
- Faster symbolic derivative than sympy
- More robust conversion from symbolic expression to numeric function

Sympy (https://www.sympy.org/en/index.html)
- Symbolic variable manipulation
