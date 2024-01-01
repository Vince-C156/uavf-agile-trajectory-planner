from pydrake.all import BasicVector, PortDataType, Simulator
import numpy as np
from rotation_utils import quaternion_to_dcm, quaternion_multiply
from pydrake.systems.framework import LeafSystem_
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.autodiffutils import AutoDiffXd
from pydrake.symbolic import Expression
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix
from scipy.spatial.transform import Rotation as R

class quad_dynamics:

    #parameters from cpc paper
    g = np.array([0, 0, -9.81])  # Gravity vector
    m = 0.76  # Mass of the quadrotor [kg] Estimate from sheets [12.578lb --> 5.70528483kg]
    J = np.diag([3, 3, 5])  # Inertia matrix [ESTIMATE]
    l = 0.17  # Length of the quadrotor arm [m] measurement from cad model
    T_max = 16  # Maximum thrust, 40kg --> 40kg * 9.81m/s^2 = 392.4N --> convervative estimate of 270N
    v_max = 42.8224  # Maximum velocity (60mph --> 26.8224 m/s)
    c_tau = 0.01  # Constant related to the torque produced by the aerodynamic drag on the rotors 2.19% of the thrust


    def calculate_dynamics(self, t, state, u):

        # Extract the state and inputs from the context
        p = state[:3]
        q = state[3:7]
        v = state[7:10]
        w = state[10:]

        quat_obj = R.from_quat(q) #[qx qy qz qw]
        R = quat_obj.as_matrix()
        
        # Calculate the drag term
        Cd_term = np.sqrt(4 * (self.T_max/self.m)**2 - ( (self.g[2]**2) / self.v_max))
        
        # Equation 12: Translational dynamics
        dp = v @ np.diag([1, 1, -1])
        
        # Equation 13: Gravitational acceleration with rotation matrix and thrust
        dv = self.g + 1/self.m * np.dot(R, np.array([0, 0, np.sum(u)])) - Cd_term * v
        dv = dv @ np.diag([1, 1, -1])
        
        # Equation 14: Quaternion dynamics
        q_dot = 0.5 * quaternion_multiply(q, np.array([0, w[0], w[1], w[2]]))
        
        # Equation 15: Rotational dynamics
        t_hat = self.calculate_torque(u)  #t_hat is the torque vector defined in the paper
        dw = np.dot(np.linalg.inv(self.J), t_hat - np.cross(w, np.dot(self.J, w)))
        
        # Return the derivatives of the state

        state_dot = np.concatenate([dp, q_dot, dv, dw])
        return state_dot

    def calculate_torque(self, u):
        """
        Calculate the torque vector from the given motor thrusts.
        
        Parameters:
        u (list): The input vector containing the motor thrusts [T1, T2, T3, T4].
        l (float): The arm length of the quadrotor.
        c_tau (float): The constant related to the torque produced by the aerodynamic drag on the rotors.
        
        Returns:
        numpy.ndarray: The torque vector τ.
        """
        T1, T2, T3, T4 = u
        tau_x = self.l / np.sqrt(2) * (T1 + T2 - T3 - T4)
        tau_y = self.l / np.sqrt(2) * (-T1 + T2 + T3 - T4)
        tau_z = self.c_tau * (T1 - T2 + T3 - T4)
        
        return np.array([tau_x, tau_y, tau_z])
    

@TemplateSystem.define("CompQuad_")
def CompQuad_(T):

    class Impl(LeafSystem_[T]):
        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter=converter)
            self.g = np.array([0, 0, -9.81])
            self.m = 5.8 #kg
            self.L = 0.38 #m
            self.J = np.diag([3, 3, 5]) 
            self.kF = 0.01 #thrust coefficient
            self.kM = 0.01 #torque from drag coefficient

            self.v_max = 22.352 #m/s

            # Declare input ports
            self.DeclareVectorInputPort("u", 4)

            # Declare the continuous state of the system
            state_index = self.DeclareContinuousState(13)

            # Declare output port
            self.DeclareStateOutputPort('state', state_index)

        def DoCalcTimeDerivatives(self, context, derivatives):
            # Extract the state and inputs
            state = context.get_continuous_state_vector().CopyToVector()
            u = self.get_input_port(0).Eval(context)

            p = state[:3]  # position
            q = state[3:7]  # orientation (quaternion)
            v = state[7:10]  # velocity
            w = state[10:]  # angular velocity

            # Convert the quaternion to a rotation matrix
            quat = np.array([q[0], q[1], q[2], q[3]]) #[qx qy qz qw]

            w = np.array([w[0], w[1], w[2]])
            Rot_matrix = quaternion_to_dcm(quat)

            # Calculate the drag term
            Cd_term = np.sqrt(4 * (self.kF/self.m)**2.0 - ( (self.g[2]**2) / self.v_max))
            
            # Translational dynamics
            dp = np.array(v)
            dp = dp * np.array([1, 1, -1])
            
            # Gravitational acceleration with rotation matrix and thrust
            dv = self.g[2] + ((1/self.m) * (Rot_matrix @ np.array([0, 0, np.sum(u)]).T) ) - (Cd_term * v)
            dv = dv * np.array([1, 1, -1])
            
            # Quaternion dynamics
            pure_quaternion = np.array([0, w[0], w[1], w[2]])

            q_prod = quaternion_multiply(quat, pure_quaternion)
            q_dot = q_prod * 0.5

            # Rotational dynamics
            tau = self.calculate_torque(u)
            J_inv = np.linalg.inv(self.J)
            dw = J_inv @ (tau - np.cross(w, (self.J @ w)) )
            state_dot = np.concatenate([dp, q_dot, dv, dw])

            derivatives.get_mutable_vector().SetFromVector(state_dot)

        def calculate_torque(self, u):
            T1, T2, T3, T4 = u
            tau_x = self.L / (np.sqrt(2) * (T1 + T2 - T3 - T4))
            tau_y = self.L / (np.sqrt(2) * (-T1 + T2 + T3 - T4))
            tau_z = self.kM * (T1 - T2 + T3 - T4)
            return np.array([tau_x, tau_y, tau_z])
        
        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)
    return Impl


'''
parameters from comp quad estimates
    g = np.array([0, 0, 9.81])  # Gravity vector
    m = 5.70528483  # Mass of the quadrotor [kg] Estimate from sheets [12.578lb --> 5.70528483kg]
    J = np.diag([0.01, 0.01, 0.02])  # Inertia matrix [ESTIMATE]
    l = 0.38  # Length of the quadrotor arm [m] measurement from cad model
    T_max = 270  # Maximum thrust, 40kg --> 40kg * 9.81m/s^2 = 392.4N --> convervative estimate of 270N
    v_max = 26.8224  # Maximum velocity (60mph --> 26.8224 m/s)
    c_tau = 0.0219  # Constant related to the torque produced by the aerodynamic drag on the rotors 2.19% of the thrust
'''

'''
class QuadrotorPlant(LeafSystem):

    g = np.array([0, 0, -9.81])  # Gravity vector
    m = 5.70528483  # Mass of the quadrotor [kg] Estimate from sheets [12.578lb --> 5.70528483kg]
    J = np.diag([0.01, 0.01, 0.02])  # Inertia matrix [ESTIMATE]
    l = 0.38  # Length of the quadrotor arm [m] measurement from cad model
    T_max = 270  # Maximum thrust, 40kg --> 40kg * 9.81m/s^2 = 392.4N --> convervative estimate of 270N
    v_max = 26.8224  # Maximum velocity (60mph --> 26.8224 m/s)
    c_tau = 0.0219  # Constant related to the torque produced by the aerodynamic drag on the rotors 2.19% of the thrust

    def __init__(self):
        LeafSystem.__init__(self)
        
        # Define the state vector: [p, q, v, ω]
        self.state = self.DeclareContinuousState(13)  # 3 for p, 4 for q, 3 for v, 3 for ω
        
        # Define the input port for thrust and torques
        self._u_port = self.DeclareVectorInputPort("u", BasicVector(4))
        
        # Define an output port if needed, for example, the full state
        self.DeclareVectorOutputPort("state", BasicVector(13), self.CopyStateOut)
        
    def DoCalcTimeDerivatives(self, context, derivatives):
        # Extract the state and inputs from the context
        p = context.get_continuous_state_vector().CopyToVector()[:3]
        q = context.get_continuous_state_vector().CopyToVector()[3:7]
        v = context.get_continuous_state_vector().CopyToVector()[7:10]
        w = context.get_continuous_state_vector().CopyToVector()[10:]
        u = self._u_port.Eval(context)
        
        # Calculate derivatives of p, q, v, and ω using the provided equations
        # Dynamics from equations in UAV-Forge-Agile-Control paper
        
        dp, dq, dv, dw = self.calculate_dynamics(p, q, v, w, u)
        
        # Write the calculated derivatives back to the derivatives vector
        derivatives.get_mutable_vector().SetFromVector(np.concatenate([dp, dq, dv, dw]))

        next_p = p + dp
        next_q = q + dq
        next_v = v + dv
        next_w = w + dw

        #context.get_continuous_state_vector().SetFromVector(np.concatenate([next_p, next_q, next_v, next_w]))
        

    def calculate_dynamics(self, p, q, v, w, u):
        # Convert the quaternion to a rotation matrix [Direct Cosine Matrix]
        quat_obj = Quaternion(q) # quaternion is [qw, qx, qy, qz]
        R = quat_obj.to_DCM()
        
        # Calculate the drag term
        Cd_term = np.sqrt(4 * (self.T_max/self.m)**2 - ( (self.g[2]**2) / self.v_max))
        
        # Equation 12: Translational dynamics
        dp = v
        
        # Equation 13: Gravitational acceleration with rotation matrix and thrust
        dv = self.g + 1/self.m * np.dot(R, np.array([0, 0, u[0]])) - Cd_term * v
        
        # Equation 14: Quaternion dynamics
        q_dot = 0.5 * (quat_obj.mult_L() @ np.array([0, w[0], w[1], w[2]]))  # This needs proper quaternion multiplication
        
        # Equation 15: Rotational dynamics
        t_hat = self.calculate_torque(u)  #t_hat is the torque vector defined in the paper
        dw = np.dot(np.linalg.inv(self.J), t_hat - np.cross(w, np.dot(self.J, w)))
        
        # Return the derivatives of the state
        return dp, q_dot, dv, dw

    def calculate_torque(self, u):
        """
        Calculate the torque vector from the given motor thrusts.
        
        Parameters:
        u (list): The input vector containing the motor thrusts [T1, T2, T3, T4].
        l (float): The arm length of the quadrotor.
        c_tau (float): The constant related to the torque produced by the aerodynamic drag on the rotors.
        
        Returns:
        numpy.ndarray: The torque vector τ.
        """
        T1, T2, T3, T4 = u
        tau_x = self.l / np.sqrt(2) * (T1 + T2 - T3 - T4)
        tau_y = self.l / np.sqrt(2) * (-T1 + T2 + T3 - T4)
        tau_z = self.c_tau * (T1 - T2 + T3 - T4)
        
        return np.array([tau_x, tau_y, tau_z])

    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)

    def print_params(self):
        print('='*40)
        print("Quadrotor Parameters:")
        print(f"m = {self.m} | mass of the quadrotor [kg]")
        print(f"J = {self.J} | inertia matrix")
        print('----------------------------------------')
        print(f"g = {self.g} | gravity vector [m/s^2]")
        print(f"l = {self.l} | length of the quadrotor arm [m]")
        print(f"T_max = {self.T_max} | maximum thrust placeholders")
        print(f"v_max = {self.v_max} | maximum velocity placeholder for drag equation")
        print(f"c_tau = {self.c_tau} | coefficient related to the torque produced by the aerodynamic drag on the rotors")
        print('='*40)
'''