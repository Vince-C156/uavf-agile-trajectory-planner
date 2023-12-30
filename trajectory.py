import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import Simulator, DiagramBuilder, MathematicalProgram, IpoptSolver, LeafSystem, eq
from uavf_types import waypoints
from dynamics import quad_dynamics
from scipy.integrate import RK45, solve_ivp
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import Integrator
import time
from datetime import datetime
#CPC Trajectory Optimization

class CPC:

    def __init__(self, dyn_plant : quad_dynamics, x0 : np.array, u_max : float, waypoints : waypoints) -> None:
        self.dynamics = dyn_plant
        self.x0 = x0
        self.u_max = u_max
        self.u_min = 0

        self.waypoints_object = waypoints
        self.waypoints_ned = self.waypoints_object.waypoints_local
        self.N_waypoints = len(self.waypoints_ned)

        self.d_tol = 0.1 #Distance tolerance to waypoint [m]

        self.prog = MathematicalProgram()
        self.solver = IpoptSolver()

        self.print_params()

    def set_up_variables(self, NPW) -> None:

        N = self.N_waypoints * NPW #Number of nodes = number of waypoints * number of nodes per waypoint

        #Define variables
        t_f = self.prog.NewContinuousVariables(1, "t_f") #Total time / final time

        self.prog.SetInitialGuess(t_f, [N])
        dt = N / t_f[0] #Time step = number of nodes / total time

        #Define state variables and input variables
        self.states = self.prog.NewContinuousVariables(N, 13, "x")
        self.inputs = self.prog.NewContinuousVariables(N - 1, 4, "u")

        #Inital state constraint

        print(f'x0: {self.x0}')
        print(f'x0 shape: {self.x0.shape}')
        inital_condition = self.prog.AddBoundingBoxConstraint(self.x0.T, self.x0.T, self.states[0, :])
        inital_condition.evaluator().set_description("Initial Condition")

        #Dynamics constraint
        for i in range(N-1):
            #Input box constraint
            actuator_limits = self.prog.AddBoundingBoxConstraint(self.u_min, self.u_max, self.inputs[i, :])
            actuator_limits.evaluator().set_description(f"Actuator_Limits_{i}")

            # Calculate the dynamics using your calculate_dynamics function
            dynamics = self.dynamics.calculate_dynamics(0, self.states[i, :], self.inputs[i, :])
            dynamics_constraint = self.prog.AddConstraint(
                eq(self.states[i + 1, :], self.states[i, :] + dynamics * dt)
            )

            dynamics_constraint.evaluator().set_description(f"Dynamics_Constraint_{i}")

        #For each waypoint, define variables
        for i in range(self.N_waypoints):

            lambda_prog = self.prog.NewContinuousVariables(N, f"lambda_{i}")
            slack_prog = self.prog.NewContinuousVariables(N, f"slack_{i}")
            mu_prog = self.prog.NewContinuousVariables(N, f"mu_{i}")

            if i == 0:

                #Inital value for progress constraint
                inital_prog = self.prog.AddBoundingBoxConstraint(1, 1, lambda_prog[0])
                inital_prog.evaluator().set_description(f"Initial_Progress_{i}")

                #Final value for progress constraint
                final_prog = self.prog.AddBoundingBoxConstraint(0, 0, lambda_prog[N - 1])
                final_prog.evaluator().set_description(f"Final_Progress_{i}")

                #Slack variable box constraint
                slack_box = self.prog.AddBoundingBoxConstraint(0, self.d_tol, slack_prog[:])
                slack_box.evaluator().set_description(f"Slack_Box_{i}")

            #For each waypoint at every step/node, define variables
            for j in range(N - 1):
                mu_prog_equality = self.prog.AddLinearConstraint(mu_prog[j] >= 0)
                #Mu variable constraints
                mu_constraint = self.prog.AddConstraint(
                    mu_prog[j] <= 1
                )

                mu_next_constraint = self.prog.AddConstraint(
                    mu_prog[j + 1] - mu_prog[j] <= 0
                )

                complementary_progress_constraint = self.prog.AddConstraint(
                    self.complementary_constraint(
                    self.states[j, :3],
                    self.waypoints_ned[i, :],
                    slack_prog[j],
                    mu_prog[j]
                    ) == 0
                )

                complementary_progress_constraint.evaluator().set_description(f"Complementary_Progress_{i}_{j}")

                #Progress variable evolution constraint
                progress_constraint = self.prog.AddConstraint(
                    lambda_prog[j + 1] - lambda_prog[j] + mu_prog[j] == 0
                )

                progress_constraint.evaluator().set_description(f"Progress_Constraint_{i}_{j}")

                self.prog.AddConstraint(
                    lambda_prog[j+1] - lambda_prog[j] <= 0
                )

        time_cost = self.prog.AddCost(t_f[0])
        time_cost.evaluator().set_description("Time_Cost")


    def complementary_constraint(self, p, p_wp, slack_var, mu):
        
        #Distance to waypoint
        d = np.linalg.norm(p - p_wp)

        #Complementary constraint
        value = mu * (d - slack_var)

        return value


    def solve(self, NPW=30):
        self.set_up_variables(NPW)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

        result = self.solver.Solve(self.prog)
        print("Success? ", result.is_success())
        print(result.get_solution_result())

        print("x* = ", result.GetSolution(self.states))
        print("Solver is ", result.get_solver_id().name())
        print("Ipopt solver status: ", result.get_solver_details().status,
            ", meaning ", result.get_solver_details().ConvertStatusToString())
        
        end_time = datetime.now()

        print(f"Time to solve: {end_time - now}")
        

    def print_params(self):
        print("=========================================")
        print("CPC Trajectory Optimization")
        print(f'x0: {self.x0}')
        print(f'u_max: {self.u_max}')
        print(f'u_min: {self.u_min}')
        print(f'solver: {self.solver}')
        print(f'N waypoints: {len(self.waypoints_ned)}')
        print("=========================================")

    def print_prog(self):
        print(self.prog)

