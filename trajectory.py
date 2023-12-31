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
from pydrake.solvers import CommonSolverOption, SolverOptions, SnoptSolver
from integration_utils import runge_kutta_step
import threading
#CPC Trajectory Optimization

class CPC:

    def __init__(self, dyn_plant : quad_dynamics, x0 : np.array, u_max : float, waypoints : waypoints) -> None:
        self.dynamics = dyn_plant
        self.x0 = x0
        self.u_max = u_max
        self.u_min = 1e-6
        self.v_max = 26 #26.8224 #Maximum velocity (60mph --> 26.8224 m/s)

        self.waypoints_object = waypoints
        self.waypoints_ned = self.waypoints_object.waypoints_local
        self.N_waypoints = len(self.waypoints_ned)

        self.d_tol = 6.096 #Distance tolerance to waypoint [m]

        self.prog = MathematicalProgram()
        self.solver = IpoptSolver()
        #self.solver = SnoptSolver()

        self.x_solution = None
        self.u_solution = None

        self.final_time_solution = None

        self.progress_variables = None

        self.wp_var_dict = {}
        self.print_params()

    def distance_waypoints(self):
        distance = 0
        for i in range(len(self.waypoints_ned) - 1):
            distance += np.linalg.norm(self.waypoints_ned[i] - self.waypoints_ned[i + 1])

        to_first_waypoint = np.linalg.norm(self.waypoints_ned[0] - self.x0[:3])
        distance += to_first_waypoint
        return distance
    
    def set_up_variables(self, NPW) -> None:

        N = self.N_waypoints * NPW #Number of nodes = number of waypoints * number of nodes per waypoint

        #Define variables
        self.t_f = self.prog.NewContinuousVariables(1, "t_f") #Total time / final time

        #Set initial guess for total time
        self.distance_total = self.distance_waypoints() #returns total distance between waypoints and take off point
        self.final_time_guess = self.distance_total / self.v_max #guess final time as distance / max velocity
        self.prog.SetInitialGuess(self.t_f, [self.final_time_guess * 2.5])

        self.dt = N / self.t_f[0] #Time step = number of nodes / total time

        #Define state variables and input variables
        self.states = self.prog.NewContinuousVariables(N, 13, "x")
        self.inputs = self.prog.NewContinuousVariables(N - 1, 4, "u")

        #Inital state constraint

        print(f'x0: {self.x0}')
        print(f'x0 shape: {self.x0.shape}')
        inital_condition = self.prog.AddBoundingBoxConstraint(self.x0.T, self.x0.T, self.states[0, :])
        inital_condition.evaluator().set_description("Initial Condition")

        thread_dynamics = threading.Thread(target=self.set_dynamics_constraint, args=(N,), name="Dynamics_Constraint_Thread")
        thread_box_constraints = threading.Thread(target=self.set_bounding_box_constraints, args=(N,), name="Box_Constraint_Thread")
        thread_box_constraints.start()
        thread_dynamics.start()

        thread_box_constraints.join()


        self.progress_variables = []
        #For each waypoint, define variables
        for i in range(self.N_waypoints):

            lambda_prog = self.prog.NewContinuousVariables(N, f"lambda_{i}")
            slack_prog = self.prog.NewContinuousVariables(N, f"slack_{i}")
            mu_prog = self.prog.NewContinuousVariables(N, f"mu_{i}")

            self.prog.AddBoundingBoxConstraint(0, 1, mu_prog[:])

            variables_dict = {'lambda': lambda_prog, 'slack': slack_prog, 'mu': mu_prog}
            self.progress_variables.append(lambda_prog)

            #Inital value for progress constraint
            inital_prog = self.prog.AddBoundingBoxConstraint(1, 1, lambda_prog[0])
            inital_prog.evaluator().set_description(f"Initial_Progress_{i}")

            #Final value for progress constraint
            final_prog = self.prog.AddBoundingBoxConstraint(0, 0, lambda_prog[-1])
            final_prog.evaluator().set_description(f"Final_Progress_{i}")

            #Slack variable box constraint
            slack_box = self.prog.AddBoundingBoxConstraint(0, self.d_tol**2.0, slack_prog[:])
            slack_box.evaluator().set_description(f"Slack_Box_{i}")

            #For each waypoint at every step/node, define variables
            for j in range(N - 1):

                self.prog.SetInitialGuess(mu_prog[j], 0)

                if j != 0:
                    self.prog.SetInitialGuess(lambda_prog[j], 1)

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

                #lambda_prog[j] - mu_prog[j] = lambda_prog[j + 1]
                progress_constraint = self.prog.AddConstraint(
                    eq(lambda_prog[j + 1], [lambda_prog[j] - mu_prog[j]])
                )

                progress_constraint.evaluator().set_description(f"Progress_Constraint_{i}_{j}")

            self.wp_var_dict.update({f'wp_{i}' : variables_dict})

        thread_dynamics.join()
        #Sequence constraints
        self.mu_sequence_constraint()
        self.lambda_sequence_constraint()

        #Time cost to minimize
        time_cost = self.prog.AddCost(self.t_f[0]**2.0)
        time_cost.evaluator().set_description("Time_Cost")


    def set_bounding_box_constraints(self, N):
        for i in range(N-1):
            #Velocity constraint
            velocities = self.states[i+1, 7:10]
            velocity_constraint = self.prog.AddBoundingBoxConstraint(-self.v_max, self.v_max, velocities[:])

            #Body rates constraint
            body_rates = self.states[i+1, 10:]
            body_rates_constraint = self.prog.AddBoundingBoxConstraint(-10, 10, body_rates[:]) #-10 to 10 rad/s constraint on rotational rates w

            #velocity_constraint = self.prog.AddConstraint(np.linalg.norm(self.states[i, 7:10]) - self.v_max <= 0)
            #Input box constraint
            actuator_limits = self.prog.AddBoundingBoxConstraint(self.u_min, self.u_max, self.inputs[i, :])
            actuator_limits.evaluator().set_description(f"Actuator_Limits_{i}")

        print("Bounding box constraints added")

    def set_dynamics_constraint(self, N):

        for i in range(N-1):
            #Take state derivative function and integrate with 4th order Runge-Kutta for next state
            next_state = runge_kutta_step(f=self.dynamics.calculate_dynamics, y0=self.states[i, :], t0=1, dt=self.dt, thrust=self.inputs[i, :])[1]

            dynamics_constraint = self.prog.AddConstraint(
                eq(self.states[i + 1, :], next_state)
            )

            #dynamics_relaxed_constraint = self.prog.AddConstraint(self.states[i + 1, :] - next_state <= 0)

            dynamics_constraint.evaluator().set_description(f"Dynamics_Constraint_{i}")

        print("Dynamics constraint added")

    def lambda_sequence_constraint(self):
        threads = []

        for i in range(self.N_waypoints - 1):
            threads.append(threading.Thread(target=self.subsequent_lambda_constraint, args=(i, i+1), name=f"Thread_{i}_{i+1}"))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()

        print("All threads complete, lambda sequence constraint added")

    def subsequent_lambda_constraint(self, lte_j, gte_j):
        lte_key = f'wp_{lte_j}'
        gte_key = f'wp_{gte_j}'

        lambda_hat1 = self.wp_var_dict[lte_key]['lambda']
        lambda_hat2 = self.wp_var_dict[gte_key]['lambda']

        for i in range(len(lambda_hat1)):
            sequence_constraint = self.prog.AddConstraint(
                lambda_hat1[i] - lambda_hat2[i] <= 0
            )


    def mu_sequence_constraint(self):
        threads = []

        for i in range(self.N_waypoints - 1):
            threads.append(threading.Thread(target=self.subsequent_mu_constraint, args=(i, i+1), name=f"Thread_{i}_{i+1}"))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()

        print("All threads complete, mu sequence constraint added")

    def subsequent_mu_constraint(self, lte_j, gte_j):
        
        lte_key = f'wp_{lte_j}'
        gte_key = f'wp_{gte_j}'

        mu_hat1 = self.wp_var_dict[lte_key]['mu']
        mu_hat2 = self.wp_var_dict[gte_key]['mu']

        for i in range(len(mu_hat1)):

            sequence_constraint = self.prog.AddConstraint(
                mu_hat1[i] - mu_hat2[i] <= 0
            )


    def velocity_constraint(self, state_vector):
        velocity_vector = state_vector[7:10]
        value = np.linalg.norm(velocity_vector)
        return value
    
    def complementary_constraint(self, p, p_wp, slack_var, mu):
        
        #Distance to waypoint
        d = np.linalg.norm(p - p_wp)

        #Complementary constraint
        value = mu * (d**2.0 - slack_var)

        return value


    def solve(self, NPW=100):
        self.set_up_variables(NPW)

        print(f'Number of decision variables: {self.prog.num_vars()}')
        print(f'Number of nodes: {NPW*self.N_waypoints}')
        print(f'Inital guess for final time: {self.final_time_guess} [S]')
        print(f'Distance to travel: {self.distance_total} [M]')
        print(f'Waypoint Tol: {self.d_tol} [M]')

        filename = "solver_verbose/debug.txt"
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintFileName, filename)
        #solver_options.SetOption(IpoptSolver().solver_id(), "max_iter", 400)
        #solver_options.SetOption(IpoptSolver().solver_id(), "tol", 1e-3)
        #solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_tol", 1e-3)
        start_time = time.time()
        result = self.solver.Solve(self.prog, solver_options=solver_options)

        end_time = time.time()    # Record the end time
        execution_time = end_time - start_time

        print(f"Time to solve: {execution_time} [S]")
        self.x_solution = result.GetSolution(self.states)
        self.u_solution = result.GetSolution(self.inputs)
        self.final_time_solution = result.GetSolution(self.t_f)

        self.progress_solution = np.array([result.GetSolution(self.progress_variables[i]) for i in range(self.N_waypoints)])

        print("Solver is ", result.get_solver_id().name())
        print("Ipopt solver status: ", result.get_solver_details().status,
            ", meaning ", result.get_solver_details().ConvertStatusToString())

        with open(filename) as f:
            print(f.read())


        self.plot_positions()
        self.plot_progress()
        self.plot_actuation()
        infeasible_constraints = result.GetInfeasibleConstraints(self.prog)
        for c in infeasible_constraints:
            print(f"infeasible constraint: {c.evaluator().get_description()}")


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

    def plot_positions(self):

        assert self.x_solution is not None, "Must solve the program before plotting results"

        interval = self.final_time_solution / len(self.x_solution)
        time = np.arange(0, self.final_time_solution, interval)
        #Plot the results
        xs, ys, zs = self.x_solution[:, 0], self.x_solution[:, 1], self.x_solution[:, 2]

        plt.plot(time, xs, label="x", color="green")
        plt.plot(time, ys, label="y", color="red")
        plt.plot(time, zs, label="z", color="blue")

        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("Position vs Time")
        plt.legend()
        plt.show()

    def plot_actuation(self):
            
        assert self.u_solution is not None, "Must solve the program before plotting results"

        interval = self.final_time_solution / len(self.u_solution)
        time = np.arange(0, self.final_time_solution, interval)
        #Plot the results
        u1, u2, u3, u4 = self.u_solution[:, 0], self.u_solution[:, 1], self.u_solution[:, 2], self.u_solution[:, 3]

        plt.plot(time, u1, label="u1", color="green")
        plt.plot(time, u2, label="u2", color="red")
        plt.plot(time, u3, label="u3", color="blue")
        plt.plot(time, u4, label="u4", color="black")

        plt.xlabel("Time [s]")
        plt.ylabel("Actuation")
        plt.title("Actuation vs Time")
        plt.legend()
        plt.show()

    def plot_progress(self):
        assert self.progress_solution is not None, "Must solve the program before plotting results"
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        interval = self.final_time_solution / len(self.progress_solution[0])
        time = np.arange(0, self.final_time_solution, interval)

        for i in range(self.N_waypoints):
            plt.plot(time, self.progress_solution[i], label=f"Progress to Waypoint {i}", color=colors[i])

        plt.xlabel("Time [s]")
        plt.ylabel("Progress")
        plt.title("Progress vs Time")
        plt.legend()
        plt.show()


