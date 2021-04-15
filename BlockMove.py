"""

Driver script for testing direct collocation code

Authors: John D'Angelo, Shreyas Sudhaman

"""

from sympy import symbols
from sympy import sin, cos
from sympy import Matrix, lambdify

import numpy as np

from ProblemDefinition import CollocationProblem

from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from colloc_methods import *

def draw_block(x_traj, interval=3):
    """
    Animate block
    x = [x, th1, x_dot, th1_dot]
    context = [L1, M1, MP, g]
    """
    # creating a blank window
    # for the animation
    fig = plt.figure()
    axis = plt.axes(xlim=(-15, 15),
                    ylim=(-15, 15))

    line, = axis.plot([], [], 'k-o', lw=2)
    block, = axis.plot([], [], 'b-', lw=2)
    path, = axis.plot([], [], 'r-', lw=2)
    axis.axhline(y=0, color='k')

    def init():
        line.set_data([], [])
        block.set_data([], [])
        path.set_data([], [])
        return line,block, path

    def animate(i):
        x, v = x_traj[i, :]
        box = np.array([[-1,1],
                        [1,1],
                        [1,-1],
                        [-1,-1],
                        [-1,1]]).astype(float)
        box[:,0]+=x
        block.set_data(box[:,0], box[:,1])
        xp=x_traj[:i,0]
        yp=xp * 0.0
        path.set_data(xp,yp)

        return block,path

    # calling the animation function
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=x_traj.shape[0], interval=interval, blit=True)
    axis.set_aspect("equal")
    plt.show()


if __name__ == "__main__":

    colloc_method = TRAP

    # define variables
    x, v = symbols("x v")
    u = symbols("u")
    state_vars = [x, v]
    control_vars = [u]

    # Given system equations
    ode = [v, u]

    t0_ = 0
    tf_ = 5
    N_ = 10

    X_start = np.array([0, 0]) # arbitrary goal state
    X_goal = np.array([10, 0]) # arbitrary goal state

    # bounds = [[lb_x, ub_x],[lb_v, ub_v],[lb_u, ub_u]]
    u_max = 10
    bounds = [[None,None],[None,None],[-u_max, u_max]]*N_
    tspan = np.linspace(t0_, tf_, N_)

    # Define problem
    problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tspan, colloc_method)

    # Initialize optimization variables
    x0 = [X_start.tolist() + [1.0]]
    for i in range(N_ - 1):
        xnew = X_start + (X_goal - X_start) * i / N_
        x0.append(xnew.tolist() + [1.0])
    x0 = np.array(x0).ravel()

    # Problem constraints
    constr_eq = NonlinearConstraint(problem.equality_constr.eval,
                                    lb=0,
                                    ub=0,
                                    jac=problem.equality_constr.jac,
                                    hess=problem.equality_constr.hess)

    # Solve Problem
    sol_opt = minimize(problem.objective.eval,
                       x0,
                       method="trust-constr",
                       jac=problem.objective.jac,
                       hess=problem.objective.hess,
                       constraints=(constr_eq),
                       bounds=bounds,
                       options={'sparse_jacobian': True})
    print("Done")
    if sol_opt.success:
        print("Success :-)")
    else:
        print("Failure :-(")

    print("Constraint violation: ", sol_opt.constr_violation)
    print("Iterations: ", sol_opt.niter)

    V = sol_opt.x.reshape(problem.N, problem.X_dim+problem.U_dim)
    X = V[:, :problem.X_dim]
    U = V[:, problem.X_dim:problem.X_dim+problem.U_dim]

    fun = lambdify(state_vars+control_vars, Matrix(ode), 'numpy')

    if colloc_method == TRAP:
        spl = interp1d(tspan, U.ravel(), kind='linear') # linear for trapezoid method
    elif colloc_method == HERM:
        spl = interp1d(tspan, U.ravel(), kind='quadratic') # quadratic for hermite simpson method
    
    def system_eqs(t, x_t):
        U_t = np.array([float(spl(t))])
        return fun(*x_t, *U_t).ravel()

    sol_ivp = solve_ivp(system_eqs, [tspan[0],tspan[-1]], X_start, method='RK45', t_eval=np.linspace(tspan[0],tspan[-1],100))

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Collocation Points vs. Integration Results")
    axs[0].plot(tspan, X[:,0])
    axs[0].plot(sol_ivp.t, sol_ivp.y[0,:])
    axs[0].plot(tspan, X[:,1])
    axs[0].plot(sol_ivp.t, sol_ivp.y[1,:])
    axs[0].set_ylabel("State Variables")

    axs[1].plot(tspan, U[:,0])
    axs[1].plot(sol_ivp.t, np.array(spl(sol_ivp.t)))
    axs[1].set_ylabel("Control Variables")
    axs[1].set_xlabel("Time [s]")

    f_X = interp1d(tspan, X.T)
    interp_X = f_X(sol_ivp.t).T

    fig2, axs = plt.subplots(2,1)
    axs[0].set_title("Collocation Error")
    axs[0].plot(sol_ivp.t, interp_X[:,0] - sol_ivp.y[0,:])
    axs[0].set_ylabel("X [meters]")
    axs[1].plot(sol_ivp.t, interp_X[:,1] - sol_ivp.y[1,:])
    axs[1].set_ylabel("v [meters/s]")
    axs[1].set_xlabel("Time [s]")

    plt.show()

    draw_block(sol_ivp.y.T)