"""

Convenience functions for animating the motion of different systems

Authors: John D'Angelo, Shreyas Sudhaman
Date: 05/01/2021
"""

# third party imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

plt.rcParams['animation.ffmpeg_path'] = os.environ.get('FFMPEG_PATH')

def rot(box, th):
    bx = box[:,0].copy()
    by = box[:,1].copy()
    box[:,0] = np.cos(th)*bx - np.sin(th)*by
    box[:,1] = np.sin(th)*bx + np.cos(th)*by 
    return box

def shift(box, x, y):
    box[:,0] += x
    box[:,1] += y
    return box

def draw_lander(x_traj, u, interval=3, save_anim=False):
    """
    Animate lunar lander.
    x = [x, dx, y, dy, th, dth]
    u = [Flat, Fup]
    """
    # creating a blank window
    # for the animation
    fig = plt.figure()
    axis = plt.axes(xlim=(-2e3, 2e3),
                    ylim=(-500, 5e3))

    line, = axis.plot([], [], 'k-o', lw=2)
    block, = axis.plot([], [], 'b-', lw=2)
    on_thrusters, = axis.plot([], [], 'ro', markersize=5,label='thrust on')
    off_thrusters, = axis.plot([], [], 'ko', markersize=5,label='thrust off')
    path, = axis.plot([], [], 'r-', lw=2)
    th = np.linspace(0, 2*np.pi)
    rad = 5000
    x = np.cos(th)*rad
    y = np.sin(th)*rad - rad
    axis.plot(x, y, 'k')
    axis.legend()

    def init():
        line.set_data([], [])
        block.set_data([], [])
        path.set_data([], [])
        return line,block, path

    def animate(i):
        x, dx, y, dy, th, dth = x_traj[i, :]
        Fl, Ft = u[i,:]
        dim = 500
        box = np.array([[0,1],
                        [1,0],
                        [-1,0],
                        [0,1]]).astype(float) * dim
        box = rot(box, th)
        box = shift(box, x, y)
        block.set_data(box[:,0], box[:,1])
        xp=x_traj[:i,0]
        yp=x_traj[:i,2]
        path.set_data(xp,yp)

        t_on = []
        t_off = []
        if Fl > 0:
            t_off.append([-dim,0])
            t_on.append([dim,0])
        elif Fl < 0:
            t_off.append([-dim,0])
            t_on.append([dim,0])
        else:
            t_off.append([dim,0])
            t_off.append([-dim,0])

        if Ft > 1.0:
            t_on.append([0,0])
        else:
            t_off.append([0,0])
        t_on = np.array(t_on, dtype=float)
        t_on = rot(t_on, th)
        t_on = shift(t_on, x, y)
        t_off = np.array(t_off, dtype=float)
        t_off = rot(t_off, th)
        t_off = shift(t_off, x, y)
        on_thrusters.set_data(t_on[:,0], t_on[:,1])
        off_thrusters.set_data(t_off[:,0], t_off[:,1])

        return block,path,on_thrusters,off_thrusters

    # calling the animation function
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=x_traj.shape[0], interval=interval, blit=True)
    axis.set_xlabel("X [meters]")
    axis.set_ylabel("Y [meters]")
    axis.set_aspect("equal")
    axis.set_aspect("equal")
    if save_anim:
        writervideo = animation.FFMpegWriter(fps=30)
        print("save")
        anim.save('lunar_lander.mp4', writer=writervideo)
        plt.close()
    else:
        plt.show()

def draw_block(x_traj, interval=3, save_anim=False):
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
    if save_anim:
        writervideo = animation.FFMpegWriter(fps=30)
        print("save")
        anim.save('blockmove.mp4', writer=writervideo)
        plt.close()
    else:
        plt.show()

def draw_cartpole(x_traj, context, interval=3, save_anim=False):
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
    if save_anim:
        writervideo = animation.FFMpegWriter(fps=30)
        print("save")
        anim.save('cartpole.mp4', writer=writervideo)
        plt.close()
    else:
        plt.show()

def draw_double_pendulum(x_traj, context, interval=3, save_anim=False):
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
    print("save")
    if save_anim:
        writervideo = animation.FFMpegWriter(fps=30)
        anim.save('blockmove.mp4', writer=writervideo)
        plt.close()
    else:
        plt.show()
