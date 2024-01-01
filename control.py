
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from pydrake.all import (AutoDiffXd,FirstOrderTaylorApproximation, DirectCollocation, DiagramBuilder, LeafSystem, Solve, MathematicalProgram, SnoptSolver, Linearize)
from uavf_types import waypoints
import threading
#Differential Flatness Based Controller

#NLMPC based controller

#MPCC based controller

#Finite Horizon LQR for inital guess in CPC

class FiniteTimeLQR:

    def __init__(self, system : LeafSystem, x0 : np.array, u_max : float, waypoints : waypoints, NWP : int):

        self.plant = system
        self.context = self.plant.CreateDefaultContext()

        self.x0 = x0
        self.u_max = u_max

        self.waypoints_object = waypoints
        self.waypoints_ned = self.waypoints_object.waypoints_local

        self.NWP = NWP

        self.linear_systems = {}

        linearization_points = np.concatenate([[self.x0[:3]], [wp for wp in self.waypoints_ned]])

        for i, point in enumerate(linearization_points):
            A, B = self.linearize_about_hover(point)
            self.linear_systems.update({f'{i}' : (A, B)})


    def linearize_about_hover(self, position):
        non_positional_states = np.array([1, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0]) #[qw, q1, q2, q3, xdot, ydot, zdot, wx, wy, wz]
        hover_about_position = np.concatenate((position, non_positional_states))
        g = 9.81  # Gravity in m/s^2
        m = 5.8   # Mass of the quadrotor in kg (replace with your quadrotor's mass)
        weight = m * g
        thrust_per_propeller = weight / 4
        u_eq = np.full(4, thrust_per_propeller)

        self.context.SetContinuousState(hover_about_position)
        self.context.SetTime(1.0)

        self.plant.get_input_port().FixValue(self.context, u_eq)

        Affine_System = FirstOrderTaylorApproximation(self.plant, self.context)
        print(Affine_System.A())
        print(Affine_System.B())
        return Affine_System.A(), Affine_System.B()
    
    def solve_segment(self, x0, xf):

        prog = MathematicalProgram()
        solver = SnoptSolver()

        states = prog.NewContinuousVariables(self.NWP, 13, "states")
        inputs = prog.NewContinuousVariables(self.NWP - 1, 4, "inputs")

        input_bounds = prog.AddBoundingBoxConstraint([0, 0, 0, 0], [self.u_max, self.u_max, self.u_max, self.u_max], inputs[:])
        final_state_constraint = prog.AddBoundingBoxConstraint(xf, xf, states[-1, :])
        inital_state_constraint = prog.AddBoundingBoxConstraint(x0, x0, states[0, :])

        pass