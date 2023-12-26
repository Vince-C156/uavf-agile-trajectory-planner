from pydrake.all import LeafSystem, BasicVector, PortDataType, Simulator
import numpy as np
import ahrs
from ahrs import Quaternion


class QuadrotorPlant(LeafSystem):

    g = np.array([0, 0, -9.81])  # Gravity vector
    m = 1.0  # Mass of the quadrotor [kg]
    J = np.diag([0.01, 0.01, 0.02])  # Inertia matrix
    l = 0.25  # Length of the quadrotor arm [m]
    T_max = 10  # Maximum thrust, placeholder
    v_max = 10  # Maximum velocity, placeholder for drag equation
    c_tau = 0.1  # Constant related to the torque produced by the aerodynamic drag on the rotors 10% of the thrust

    def __init__(self):
        LeafSystem.__init__(self)
        
        # Define the state vector: [p, q, v, ω]
        self.DeclareContinuousState(13)  # 3 for p, 4 for q, 3 for v, 3 for ω
        
        # Define the input port for thrust and torques
        self.DeclareVectorInputPort("u", BasicVector(4))
        
        # Define an output port if needed, for example, the full state
        self.DeclareVectorOutputPort("state", BasicVector(13), self.CopyStateOut)
        
    def DoCalcTimeDerivatives(self, context, derivatives):
        # Extract the state and inputs from the context
        p = context.get_continuous_state_vector().CopyToVector()[:3]
        q = context.get_continuous_state_vector().CopyToVector()[3:7]
        v = context.get_continuous_state_vector().CopyToVector()[7:10]
        w = context.get_continuous_state_vector().CopyToVector()[10:]
        u = self.EvalVectorInput(context, 0).CopyToVector()
        
        # Calculate derivatives of p, q, v, and ω using the provided equations
        # Dynamics from equations in UAV-Forge-Agile-Control paper
        
        dp, dq, dv, dw = self.calculate_dynamics(p, q, v, w, u)
        
        # Write the calculated derivatives back to the derivatives vector
        derivatives.get_mutable_vector().SetFromVector(np.concatenate([dp, dq, dv, dw]))
        

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
        q_dot = 0.5 * (quat_obj.mult_L(np.array([0, w[0], w[1], w[2]])))  # This needs proper quaternion multiplication
        
        # Equation 15: Rotational dynamics
        t_hat = self.calculate_torque(u)  #t_hat is the torque vector defined in the paper
        dw = np.dot(np.linalg.inv(self.J), t_hat - np.cross(w, np.dot(self.J, w)))
        
        # Return the derivatives of the state
        return np.concatenate([dp, q_dot, dv, dw])

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