from typing import TypeAlias, TypeVar, Union, Any, Callable, Tuple, List
import jax.numpy as jnp
import jax
from sympy import symbols, Function
from dataclasses import dataclass
from mpmath import radians
import config
import utils
from jax.numpy.linalg import inv
from scipy.integrate import ode
from jax.numpy import sin, cos, tan, pi, sign
import matplotlib.pyplot as plt
import matplotlib
from ahrs import Quaternion


matplotlib.use('qtagg')

# Define types
# ---------------------------

coordinates : TypeAlias = list[float, float]
waypoints_global: TypeAlias = list[coordinates]
waypoints_local: TypeAlias = list[jnp.array]

@dataclass
class polynomial_trajectory:
    states : jnp.ndarray
    inputs : jnp.ndarray


class controller:
    def __init__(self, x0 : jnp.array, dt : float, waypoints : waypoints_local):

        self.x0 = x0
        self.dt = dt

        self.waypoints = waypoints

    

def nlmpc_controller(controller):
    def __init__(self, x0 : jnp.array, dt : float, waypoints : waypoints_local):
        super().__init__(x0, dt, waypoints)

        self.waypoints = waypoints

def dfbc_controller(controller):
    pass

class trajectory_optimization:
    def __init__(self, global_origin : coordinates, waypoints : waypoints_global):
        self.global_origin = global_origin
        self.waypoints_global = waypoints

        self.waypoints_local = jnp.array([self.convert_to_ned(lat, lon) for lat, lon in self.waypoints_global])

    def convert_to_ned(self, lat, lon):

        origin_lat, origin_lon = self.global_origin
        # Earth radius in meters
        R = 6371000  # Approximate radius of the Earth

        # Convert latitude and longitude from degrees to radians
        lat = jnp.radians(lat)
        lon = jnp.radians(lon)
        origin_lat = jnp.radians(origin_lat)
        origin_lon = jnp.radians(origin_lon)

        # Calculate the NED coordinates
        d_lat = lat - origin_lat
        d_lon = lon - origin_lon

        # Calculate the North, East, and Down components
        ned_north = R * d_lat
        ned_east = R * jnp.cos(origin_lat) * d_lon
        ned_down = 0  # You may adjust this if needed for altitude differences

        return jnp.array([ned_north, ned_east, ned_down])
    
    def verbose(self):
        print("Global origin: {}".format(self.global_origin))
        print("Global waypoints: {}".format(self.waypoints_global))
        print("Local waypoints: {}".format(self.waypoints_local))

    def plot_waypoints(self):
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(self.waypoints_local[1:,0], self.waypoints_local[1:,1], self.waypoints_local[1:,2], 'gray')

        ax.scatter(self.waypoints_local[0,0], self.waypoints_local[0,1], self.waypoints_local[0,2], 'red')

        ax.set_zlim(0, 100)
        plt.show()

class cpc_trajectory_optimization(trajectory_optimization):
    def __init__(self,  global_origin : coordinates, waypoints : waypoints_global, x0 : jnp.array, dt : float, T : float):
        super().__init__(global_origin, waypoints)
        self.x0 = x0
        self.dt = dt
        self.T = T

        super().verbose()
        super().plot_waypoints()

    def optimize(self):
        pass


class dynamics_quad:
    def __init__(self, params : dict):
        self.params = params

        
        # Command for initial stable hover
        # ---------------------------
        ini_hover = init_cmd(self.params)
        self.params["FF"] = ini_hover[0]         # Feed-Forward Command for Hover
        self.params["w_hover"] = ini_hover[1]    # Motor Speed for Hover
        self.params["thr_hover"] = ini_hover[2]  # Motor Thrust for Hover  
        self.thr = jnp.ones(4)*ini_hover[2]
        self.tor = jnp.ones(4)*ini_hover[3]

        # Initial State
        # ---------------------------
        self.state = init_state(self.params)

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = jnp.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        self.vel_dot = jnp.zeros(3)
        self.omega_dot = jnp.zeros(3)
        self.acc = jnp.zeros(3)

        self.extended_state()
        self.forces()

        # Set Integrator
        # ---------------------------
        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, 0)


    def extended_state(self):

        # Rotation Matrix of current state (Direct Cosine Matrix)
        self.dcm = utils.quat2Dcm(self.quat)

        # Euler angles of current state
        YPR = utils.quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1] # flip YPR so that euler state = phi, theta, psi
        self.psi   = YPR[0]
        self.theta = YPR[1]
        self.phi   = YPR[2]

    
    def forces(self):
        
        # Rotor thrusts and torques
        self.thr = self.params["kTh"]*self.wMotor*self.wMotor
        self.tor = self.params["kTo"]*self.wMotor*self.wMotor

    def state_dot(self, t, state, cmd, wind):

        # Import Params
        # ---------------------------    
        mB   = self.params["mB"]
        g    = self.params["g"]
        dxm  = self.params["dxm"]
        dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = self.params["Cd"]
        
        kTh  = self.params["kTh"]
        kTo  = self.params["kTo"]
        tau  = self.params["tau"]
        kp   = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (config.usePrecession):
            uP = 1
        else:
            uP = 0
    
        # Import State Vector
        # ---------------------------  
        x      = state[0]
        y      = state[1]
        z      = state[2]
        q0     = state[3]
        q1     = state[4]
        q2     = state[5]
        q3     = state[6]
        xdot   = state[7]
        ydot   = state[8]
        zdot   = state[9]
        p      = state[10]
        q      = state[11]
        r      = state[12]
        wM1    = state[13]
        wdotM1 = state[14]
        wM2    = state[15]
        wdotM2 = state[16]
        wM3    = state[17]
        wdotM3 = state[18]
        wM4    = state[19]
        wdotM4 = state[20]

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        
        uMotor = cmd
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*uMotor[0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*uMotor[1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*uMotor[2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*uMotor[3])/(tau**2)
    
        wMotor = jnp.array([wM1, wM2, wM3, wM4])
        wMotor = jnp.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh*wMotor*wMotor
        torque = kTo*wMotor*wMotor
    
        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # Wind Model
        # ---------------------------
        [velW, qW1, qW2] = wind.randomWind(t)
        # velW = 0

        # velW = 5          # m/s
        # qW1 = 0*deg2rad    # Wind heading
        # qW2 = 60*deg2rad     # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU)
    
        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        if (config.orient == "NED"):
            DynamicsDot = jnp.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + ( ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz]])
        elif (config.orient == "ENU"):
            DynamicsDot = jnp.array([
                [                                                                                                                                   xdot],
                [                                                                                                                                   ydot],
                [                                                                                                                                   zdot],
                [                                                                                                        -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [                                                                                                         0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [                                                                                                         0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [                                                                                                        -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [     (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [     (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB],
                [ (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + (ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],
                [                                    ((IByy - IBzz)*q*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx], # uP activates or deactivates the use of gyroscopic precession.
                [                                    ((IBzz - IBxx)*p*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy], # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                [                                                                               ((IBxx - IBzz)*p*q + TorM1 - TorM2 + TorM3 - TorM4)/IBzz]])
    
    
        # State Derivative Vector
        # ---------------------------
        sdot     = jnp.zeros([21])
        sdot[0]  = DynamicsDot[0]
        sdot[1]  = DynamicsDot[1]
        sdot[2]  = DynamicsDot[2]
        sdot[3]  = DynamicsDot[3]
        sdot[4]  = DynamicsDot[4]
        sdot[5]  = DynamicsDot[5]
        sdot[6]  = DynamicsDot[6]
        sdot[7]  = DynamicsDot[7]
        sdot[8]  = DynamicsDot[8]
        sdot[9]  = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4

        self.acc = sdot[7:10]

        return sdot


def sys_params():
    mB  = 1.2       # mass (kg)
    g   = 9.81      # gravity (m/s/s)
    dxm = 0.16      # arm length (m)
    dym = 0.16      # arm length (m)
    dzm = 0.05      # motor height (m)
    IB  = jnp.array([[0.0123, 0,      0     ],
                    [0,      0.0123, 0     ],
                    [0,      0,      0.0224]]) # Inertial tensor (kg*m^2)
    IRzz = 2.7e-5   # Rotor moment of inertia (kg*m^2)


    params = {}
    params["mB"]   = mB
    params["g"]    = g
    params["dxm"]  = dxm
    params["dym"]  = dym
    params["dzm"]  = dzm
    params["IB"]   = IB
    params["invI"] = inv(IB)
    params["IRzz"] = IRzz
    params["useIntergral"] = bool(False)    # Include integral gains in linear velocity control
    # params["interpYaw"] = bool(False)       # Interpolate Yaw setpoints in waypoint trajectory

    params["Cd"]         = 0.1
    params["kTh"]        = 1.076e-5 # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
    params["kTo"]        = 1.632e-7 # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
    params["mixerFM"]    = makeMixerFM(params) # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
    params["mixerFMinv"] = inv(params["mixerFM"])
    params["minThr"]     = 0.1*4    # Minimum total thrust
    params["maxThr"]     = 9.18*4   # Maximum total thrust
    params["minWmotor"]  = 75       # Minimum motor rotation speed (rad/s)
    params["maxWmotor"]  = 925      # Maximum motor rotation speed (rad/s)
    params["tau"]        = 0.015    # Value for second order system for Motor dynamics
    params["kp"]         = 1.0      # Value for second order system for Motor dynamics
    params["damp"]       = 1.0      # Value for second order system for Motor dynamics
    
    params["motorc1"]    = 8.49     # w (rad/s) = cmd*c1 + c0 (cmd in %)
    params["motorc0"]    = 74.7
    params["motordeadband"] = 1   
    # params["ifexpo"] = bool(False)
    # if params["ifexpo"]:
    #     params["maxCmd"] = 100      # cmd (%) min and max
    #     params["minCmd"] = 0.01
    # else:
    #     params["maxCmd"] = 100
    #     params["minCmd"] = 1
    
    return params

def makeMixerFM(params):
    dxm = params["dxm"]
    dym = params["dym"]
    kTh = params["kTh"]
    kTo = params["kTo"] 

    # Motor 1 is front left, then clockwise numbering.
    # A mixer like this one allows to find the exact RPM of each motor 
    # given a desired thrust and desired moments.
    # Inspiration for this mixer (or coefficient matrix) and how it is used : 
    # https://link.springer.com/article/10.1007/s13369-017-2433-2 (https://sci-hub.tw/10.1007/s13369-017-2433-2)
    if (config.orient == "NED"):
        mixerFM = jnp.array([[    kTh,      kTh,      kTh,      kTh],
                            [dym*kTh, -dym*kTh,  -dym*kTh, dym*kTh],
                            [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                            [   -kTo,      kTo,     -kTo,      kTo]])
    elif (config.orient == "ENU"):
        mixerFM = jnp.array([[     kTh,      kTh,      kTh,     kTh],
                            [ dym*kTh, -dym*kTh, -dym*kTh, dym*kTh],
                            [-dxm*kTh, -dxm*kTh,  dxm*kTh, dxm*kTh],
                            [     kTo,     -kTo,      kTo,    -kTo]])
    
    
    return mixerFM

def init_cmd(params):
    mB = params["mB"]
    g = params["g"]
    kTh = params["kTh"]
    kTo = params["kTo"]
    c1 = params["motorc1"]
    c0 = params["motorc0"]
    
    # w = cmd*c1 + c0   and   m*g/4 = kTh*w^2   and   torque = kTo*w^2
    thr_hover = mB*g/4.0
    w_hover   = jnp.sqrt(thr_hover/kTh)
    tor_hover = kTo*w_hover*w_hover
    cmd_hover = (w_hover-c0)/c1
    return [cmd_hover, w_hover, thr_hover, tor_hover]

def init_state(params):
    
    x0     = 0.  # m
    y0     = 0.  # m
    z0     = 0.  # m
    phi0   = 0.  # rad
    theta0 = 0.  # rad
    psi0   = 0.  # rad

    quat = utils.YPRToQuat(psi0, theta0, phi0)
    
    if (config.orient == "ENU"):
        z0 = -z0

    s = jnp.zeros(21)
    s[0]  = x0       # x
    s[1]  = y0       # y
    s[2]  = z0       # z
    s[3]  = quat[0]  # q0
    s[4]  = quat[1]  # q1
    s[5]  = quat[2]  # q2
    s[6]  = quat[3]  # q3
    s[7]  = 0.       # xdot
    s[8]  = 0.       # ydot
    s[9]  = 0.       # zdot
    s[10] = 0.       # p
    s[11] = 0.       # q
    s[12] = 0.       # r

    w_hover = params["w_hover"] # Hovering motor speed
    wdot_hover = 0.              # Hovering motor acc

    s[13] = w_hover
    s[14] = wdot_hover
    s[15] = w_hover
    s[16] = wdot_hover
    s[17] = w_hover
    s[18] = wdot_hover
    s[19] = w_hover
    s[20] = wdot_hover
    
    return s


import numpy as np
from pydrake.systems.framework import LeafSystem_, PortDataType
from pydrake.systems.pyplot_visualizer import PyPlotVisualizer
from pydrake.systems.scalar_conversion import TemplateSystem

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html


# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("Quadrotor3D_")
def Quadrotor3D_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter)
            # two inputs (thrust)
            self.DeclareVectorInputPort("u", 4)
            # three positions, four quaternions, three velocities, three angular velocities [x,y,z,q0,q1,q2,q3,x_dot,y_dot,z_dot,p,q,r]
            state_index = self.DeclareContinuousState(3, 4, 3, 3)
            # six outputs (full state)
            self.DeclareStateOutputPort("x", state_index)

            # parameters based on [Bouadi, Bouchoucha, Tadjine, 2007]
            self.length = 0.25  # length of rotor arm
            self.mass = 0.486  # mass of quadrotor
            self.inertia = 0.00383  # moment of inertia
            self.gravity = 9.81  # gravity

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()
            #q = x[:3]

            torque_constant = 0.01

            p = x[:3]
            q = Quaternion(x[3:7])
            v = x[7:10]
            omg = x[10:13]

            tau = np.array([[self.length / np.sqrt(2) * (u[0] + u[1]) - (u[2] + u[3])],
                            [self.length / np.sqrt(2) * (-u[0] + u[1] + u[2] - u[3])],
                            torque_constant * (u[0] - u[1] + u[2] - u[3])])

            pdot = v
            qdot = 0.5 * ( q.mult_L([0, omg[0], omg[1], omg[2]]).to_array() )
            vdot = self.gravity + (q.rotate([0, 0, np.sum(u)]) / self.mass)

            qdot = x[3:]
            qddot = np.array(
                [
                    -np.sin(q[2]) / self.mass * (u[0] + u[1]),
                    np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                    self.length / self.inertia * (u[0] - u[1]),
                ]
            )
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot))
            )

    return Impl