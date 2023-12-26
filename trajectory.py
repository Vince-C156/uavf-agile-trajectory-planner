import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import Simulator, DiagramBuilder, MathematicalProgram, IpoptSolver, LeafSystem
from uavf_types import waypoints
#CPC Trajectory Optimization

class CPC:

    def __init__(self, dyn_plant : LeafSystem, x0 : np.array, dt : float, u_max : float, u_min : float, waypoints : waypoints):
        self.dyn_plant = dyn_plant
        self.x0 = x0
        self.dt = dt
        self.u_max = u_max
        self.u_min = u_min

        self.waypoints_object = waypoints
        self.waypoints_ned = self.waypoints_object.waypoints_local

        self.prog = MathematicalProgram()
        self.solver = IpoptSolver()

        self.print_params()

    def solve(self, N):
        pass

    def print_params(self):
        print("=========================================")
        print("CPC Trajectory Optimization")
        print(f'x0: {self.x0}')
        print(f'dt: {self.dt}')
        print(f'u_max: {self.u_max}')
        print(f'u_min: {self.u_min}')
        print(f'solver: {self.solver}')
        print(f'N waypoints: {len(self.waypoints_ned)}')
        print("=========================================")
