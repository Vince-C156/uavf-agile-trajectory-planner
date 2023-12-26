import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import Simulator, DiagramBuilder, MathematicalProgram, IpoptSolver, LeafSystem
from uavf_types import waypoints
from dynamics import quad_dynamics
from scipy.integrate import RK45, solve_ivp
#CPC Trajectory Optimization

class CPC:

    def __init__(self, dyn_plant : quad_dynamics, x0 : np.array, u_max : float, u_min : float, waypoints : waypoints):
        self.dynamics = dyn_plant
        self.x0 = x0
        self.u_max = u_max
        self.u_min = u_min

        self.waypoints_object = waypoints
        self.waypoints_ned = self.waypoints_object.waypoints_local

        self.prog = MathematicalProgram()
        self.solver = IpoptSolver()

        self.print_params()

    def set_up_variables(self, N):
        # Problem variables
        self.x = []
        self.xg = []
        self.g = []
        self.lb = []
        self.ub = []
        self.J = []

        # Add variables

        #Total time
        t = self.prog.NewContinuousVariables(1, 't')
        self.x += t

        self.J = t

        #self.xg += [self.t_guess]

        self.g += [t]

        self.lb += [0.1]
        self.ub += [10]

        #inital state
        x_k = self.prog.NewContinuousVariables(13, "x_init")
        self.prog.AddBoundingBoxConstraint(self.x0, self.x0, x_k)
        self.x += x_k
        self.xg += [x_k]

        #progress variable inital
        mu_k = self.prog.NewContinuousVariables(1, "mu_init")
        self.prog.AddConstraint(mu_k == 1)
        self.x += mu_k

        for i in range(N):

            #input actuation and bounds
            uk = self.prog.NewContinuousVariables(4, f"u_{i}")
            self.prog.AddBoundingBoxConstraint(self.u_min, self.u_max, uk)

            #feedforward dynamics
            xn = solve_ivp(self.dynamics.calculate_dynamics, [0, (t/N)], x_k, method='RK45', args=(uk))

            self.x += [uk]



    def solve(self, N):
        self.set_up_variables(N)


    def print_params(self):
        print("=========================================")
        print("CPC Trajectory Optimization")
        print(f'x0: {self.x0}')
        print(f'u_max: {self.u_max}')
        print(f'u_min: {self.u_min}')
        print(f'solver: {self.solver}')
        print(f'N waypoints: {len(self.waypoints_ned)}')
        print("=========================================")
