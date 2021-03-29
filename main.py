"""

Core functionality code.
This script executes basic functions of py.

1.) setting up model equations: dX/dt

Pushed cart
--> []

Action Items:
John
a.) get model equations
b.) put in .pdf file so it can be copied over

2.) defining equations for collocation

taking the model from 1 and setting up the constraints 
and objective for the optimization. 

Obj(x,u) = 
C_eq(x,u) = ...
C_ineq(x,u) = ...

Action Items:
John --> a.) write a method that takes in the model and returns the collocation equalilty and inequality constriants
Constraints(X,U, ...) --> array of values (1 for each constraint)

Shreyas --> b.) write objective function that takes current optimization variables and returns a value

n = system state vector size
m = control input vector size
N = number of time steps

Given X, U
X = system state vector (N, n)
U = control input vector (N, m)

def Objective(X, U, ...) --> value

Block moving:

block locations from t=0 to t=tf
x(t0), x(t1), ..., x(tf)

control inputs from t=0 to t=tf
u(t0), u(t1), ..., u(tf)

move the block to a location (x_goal) but not try to hard
penalize not being at x_goal
penalize using our control input

sum up differences from x_goal
sum(sqrt((x(t) - x_goal)^2)) over t=t0 to t=tf
 
sum up control effort
sum(u(t)) over t=t0 to t=tf

Objective(x,u) = sum(sqrt((x(t) - x_goal)^2)) + sum(u(t))

minimizing Objective(x,u)

**After done with 2.), meet back up to discuss how they were implemented

3.) Setting up and running the optimization

Plug in functions into function
scipy.optimize.minimize (trust-constr)
ipopt

Action Items:
a.) feed objective function and constraint function into scipy optimize function

4.) Visualizing and validating result

optimization gives up X_col, U_col
integrate out system equations using U_col from t=0 to t=tf --> X_int
compare X_int to X_col

Actions Items:
a.) integration using ode45 or other integration method
b.) plotting with matplotlib
- plot X_int vs. X_col
- cool visual of block moving


"""