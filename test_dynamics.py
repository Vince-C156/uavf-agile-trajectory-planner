from dynamics import quad_dynamics as QuadrotorDynamics
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def enu_ned():

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the origin point
    origin = np.array([0, 0, 0])

    # Define the axis vectors for NED
    ned_x = np.array([1, 0, 0])  # North
    ned_y = np.array([0, 1, 0])  # East
    ned_z = np.array([0, 0, -1])  # Down

    # Define the axis vectors for ENU
    enu_x = np.array([0.5, 0, 0])  # East
    enu_y = np.array([0, 0.5, 0])  # North
    enu_z = np.array([0, 0, 0.5])  # Up

    # Plot the NED frame
    ax.quiver(*origin, *ned_x, color='r', label='NED - North')
    ax.quiver(*origin, *ned_y, color='g', label='NED - East')
    ax.quiver(*origin, *ned_z, color='b', label='NED - Down')

    # Plot the ENU frame
    ax.quiver(*origin, *enu_x, color='m', label='ENU - East')
    ax.quiver(*origin, *enu_y, color='y', label='ENU - North')
    ax.quiver(*origin, *enu_z, color='c', label='ENU - Up')

    # Set axis limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

def main():

    #x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #Initial state
    x0 = np.ones(13)

    XS = []
    XS.append(x0)
    my_dynamics = QuadrotorDynamics()

    inputs = np.array([26, 26, 26, 26])
    #inputs = np.array([0, 0, 0, 0])

    for i in range(100):
        xdot = my_dynamics.calculate_dynamics(0, x0, inputs)
        x0 = x0 + xdot

        XS.append(x0)
    
    print(x0)
    XS = np.array(XS)
    xs, ys, zs = XS[:, 0], XS[:, 1], XS[:, 2]
    xdots, ydots, zdots = XS[:, 7], XS[:, 8], XS[:, 9]

    fig = plt.figure()

    plt.plot(range(101), xs, label='x', color='blue')
    plt.plot(range(101), ys, label='y', color='green')
    plt.plot(range(101), zs, label='z', color='red')

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Position vs Time')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.plot(range(101), xdots, label='xdot', color='blue')
    plt.plot(range(101), ydots, label='ydot', color='green')
    plt.plot(range(101), zdots, label='zdot', color='red')
    plt.legend()
    plt.show()

    ax = plt.figure().add_subplot(111, projection='3d')

    ax.scatter(xs[0], ys[0], zs[0], label='Start', color='green')

    ax.plot(xs, ys, zs, label='Position')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    

if __name__ == "__main__":
    main()
    #enu_ned()