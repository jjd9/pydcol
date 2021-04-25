"""

Convenience functions for animating the motion of different systems

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def draw_block(x_traj, interval=3):
    """
    Animate block
    x = [x, v]
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

def draw_cartpole(x_traj, context, interval=3):
    """
    Animate cartpole
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
        x, th1, _, _ = x_traj[i, :]
        L1, _, _, _ = context
        origin = np.array([x, 0])
        end_pt = np.array([x+L1*np.sin(th1), -L1*np.cos(th1)])
        pole = np.array([origin, end_pt])

        line.set_data(pole[:, 0], pole[:, 1])
        box = np.array([[-1,1],
                        [1,1],
                        [1,-1],
                        [-1,-1],
                        [-1,1]]).astype(float)
        box[:,0]+=x
        block.set_data(box[:,0], box[:,1])
        xp=x_traj[:i,0]+L1*np.sin(x_traj[:i,1])
        yp=-L1*np.cos(x_traj[:i,1])
        path.set_data(xp,yp)

        return line,block,path

    # calling the animation function
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=x_traj.shape[0], interval=interval, blit=True)
    axis.set_aspect("equal")
    plt.show()

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
