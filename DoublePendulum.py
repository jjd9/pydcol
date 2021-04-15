"""

Driver script for testing direct collocation code

Authors: John D'Angelo, Shreyas Sudhaman

"""

from sympy import symbols
from sympy import sin, cos
from sympy import lambdify, Matrix

import numpy as np

from ProblemDefinition import CollocationProblem

from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline, interp1d

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import dill, os

from colloc_methods import *

def draw_double_pendulum(x_traj, context, interval=3):
    """
    Animate double pendulum
    x = [th1, th1_dot, th2, th2_dot]
    context = [L1, L2, M1, M2, g]
    """
    # creating a blank window
    # for the animation
    fig = plt.figure()
    axis = plt.axes(xlim=(-5, 5),
                    ylim=(-5, 5))

    line, = axis.plot([], [], 'k-o', lw=2)
    axis.axhline(y=0, color='k')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        th1, th1_dot, th2, th2_dot = x_traj[i, :]
        L1, L2, _, _, _ = context
        origin = np.array([0, 0])
        elbow_pt = L1*np.array([np.sin(th1), -np.cos(th1)])
        end_pt = elbow_pt + L2*np.array([np.sin(th2), -np.cos(th2)])
        acrobot = np.array([origin, elbow_pt, end_pt])

        line.set_data(acrobot[:, 0], acrobot[:, 1])

        return line,

    # calling the animation function
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=x_traj.shape[0], interval=interval, blit=True)
    plt.show()


if __name__ == "__main__":

    colloc_method = TRAP # HERM is really slow

    print("Initialize")
    # physical parameters
    l1 = 2.0
    l2 = 2.0
    m1 = 3.0
    m2 = 3.0
    g = 9.81

    # define variables
    theta, th_dot, phi, phi_dot = symbols("theta th_dot phi phi_dot")
    tau = symbols("tau")
    state_vars = [theta, th_dot, phi, phi_dot]
    control_vars = [tau]

    # Given system equations
    ode = [th_dot,
           (l2*(g*m1*sin(theta) + g*m2*sin(theta) - l2*m2*sin(phi - theta)*phi_dot**2) - (g*l2*m2*sin(phi) +
                                                                                          l1*l2*m2*sin(phi - theta)*th_dot**2 - tau)*cos(phi - theta))/(l1*l2*(-m1 + m2*cos(phi - theta)**2 - m2)),
           phi_dot,
           (-l2*m2*(g*m1*sin(theta) + g*m2*sin(theta) - l2*m2*sin(phi - theta)*phi_dot**2)*cos(phi - theta) + (m1 + m2)
            * (g*l2*m2*sin(phi) + l1*l2*m2*sin(phi - theta)*th_dot**2 - tau))/(l2**2*m2*(-m1 + m2*cos(phi - theta)**2 - m2))
           ]

    t0_ = 0
    tf_ = 3
    N_ = 50
    # N_ = 100

    X_start = np.array([0, 0, 0, 0], dtype=np.float) # arbitrary goal state
    X_goal = np.array([np.pi, 0, np.pi, 0], dtype=np.float) # arbitrary goal state

    # bounds
    u_max = 100
    bounds = [[-2*np.pi,2*np.pi],[None, None],[-2*np.pi,2*np.pi],[None, None],[-u_max,u_max]]*N_
    tspan = np.linspace(t0_, tf_, N_)

    # Define problem (unless we have already done this, in which case use cached problem)
    print("Define problem")
    problem = CollocationProblem(state_vars, control_vars, ode, X_start, X_goal, tspan, colloc_method)

    print("Initial conditions")

    # Initialize optimization variables
    # x0 = [X_start.tolist() + [1.0]]
    # for i in range(N_ - 1):
    #     xnew = X_start + (X_goal - X_start) * i / N_
    #     x0.append(xnew.tolist() + [1.0])
    x0 = np.random.rand(N_, X_start.size + 1)
    x0[0,:-1] = X_start.copy()
    x0[-1,:-1] = X_goal.copy()
    x0 = np.array(x0).ravel()

    print("Run optimization")

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

    # compose optimizer output
    V = sol_opt.x.reshape(problem.N, problem.X_dim+problem.U_dim)
    X = V[:, :problem.X_dim]
    U = V[:, problem.X_dim:problem.X_dim+problem.U_dim]

    # function for integrating system
    fun = lambdify(state_vars+control_vars, Matrix(ode), 'numpy')
    if colloc_method == TRAP:
        spl = interp1d(tspan, U.ravel(), kind='linear') # linear for trapezoid method
    elif colloc_method == HERM:
        spl = interp1d(tspan, U.ravel(), kind='quadratic') # quadratic for hermite simpson method

    def system_eqs(t, x_t):
        U_t = np.array([float(spl(t))])
        return fun(*x_t, *U_t).ravel()

    # integrate system with 4th order runge-kutta method
    sol_ivp = solve_ivp(system_eqs, [tspan[0],tspan[-1]], X_start, method='RK45', t_eval=np.linspace(tspan[0],tspan[-2],100))

    # plot results
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Collocation Points vs. Integration Results")
    axs[0].plot(tspan, X[:,0])
    axs[0].plot(sol_ivp.t, sol_ivp.y[0,:])
    axs[0].plot(tspan, X[:,2])
    axs[0].plot(sol_ivp.t, sol_ivp.y[2,:])
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
    axs[0].set_ylabel("q1 [rads]")
    axs[1].plot(sol_ivp.t, interp_X[:,2] - sol_ivp.y[2,:])
    axs[1].set_ylabel("q2 [rads]")
    axs[1].set_xlabel("Time [s]")

    plt.show()

    draw_double_pendulum(sol_ivp.y.T, [l1, l2, m1, m2, g])
