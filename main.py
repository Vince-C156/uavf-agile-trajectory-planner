from uavf_types import waypoints, coordinates, waypoints_global
from trajectory import CPC
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pydrake.systems.primitives import LogVectorOutput
from pydrake.systems.analysis import Simulator 
from pydrake.systems.framework import DiagramBuilder 
from dynamics import QuadrotorPlant
from dynamics import quad_dynamics as QuadrotorDynamics
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.examples import StabilizingLQRController
from pydrake.examples import QuadrotorPlant as LinearQuadrotorPlant
import pydot
import numpy as np


def main():

    # Global origin
    global_origin = coordinates([38.31510824923089, -76.54914848803199, 142]) #lat long alt[MSL FT] altitude is calculated from rules 217MSL = 75AGL so ground level is 217-75 = 142FT MSL

    # Waypoints 38.314873971866206, -76.54778747782218     38.31585045449336, -76.55363469322234    38.316793252827786, -76.55132799356905   38.316086155226266, -76.5477016471374
    wps = waypoints_global([
    coordinates([38.314873971866206, -76.54778747782218, 220.9925880461919]),
    coordinates([38.31585045449336, -76.55363469322234, 215.7410881932995]),
    coordinates([38.316793252827786, -76.55132799356905, 220.9925880461919]),
    coordinates([38.316086155226266, -76.5477016471374, 229.3553046988733]),
    ])
    # Initial state
    x = np.array([0.01, 0.01, -0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.01, 0.00, 0.00, 0.00])

    mission_waypoints = waypoints(global_origin = global_origin, waypoints = wps)

    mission_waypoints.verbose()
    mission_waypoints.plot_waypoints()



    
    builder = DiagramBuilder()

    #Quadrotor Dynamics
    quad_dynamics = builder.AddNamedSystem("Quad-Dynamics-Non-Linear", QuadrotorPlant())
    quad_context = quad_dynamics.CreateDefaultContext()
    quad_dynamics.get_input_port(0).FixValue(quad_context, [10, 10, 10, 10])

    #Log the output of the quadrotor dynamics
    logger = LogVectorOutput(quad_dynamics.get_output_port(0), builder)
    #Trajectory Optimization
    thrust_per_motor_competition = 9.07185 #N
    cpc = CPC(dyn_plant=QuadrotorDynamics(), x0=x, u_max=16, waypoints=mission_waypoints)
    cpc.solve(NPW=50)

    input()

    plant = builder.AddSystem(LinearQuadrotorPlant())

    controller = builder.AddSystem(StabilizingLQRController(plant, [0, 0, 1]))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(controller.get_output_port(0), quad_dynamics.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))


    # Build the system
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    quad_context.SetContinuousState([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    plot_system_graphviz(diagram)
    plt.savefig('quadrotor_system_diagram.png')

    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(1)

    # Plot the results.
    log = logger.FindLog(simulator.get_context())
    print(f'{log.data().transpose()}')
    #plt.figure()
    #plt.stem(log.sample_times(), log.data().transpose())
    #plt.xlabel('n')
    #plt.ylabel('y[n]')

    #quad_plant.print_params()
if __name__ == "__main__":
    main()


'''
    # Waypoints
    wps = waypoints_global([
    coordinates([38.31455389063634, -76.54800508052972, 290.9925880461919]),
    coordinates([38.31513573999766, -76.5536181046485, 285.7410881932995]),
    coordinates([38.31513573999766, -76.5517470966089, 388.86945004683116]),
    coordinates([38.31513573999766, -76.5498760885693, 299.3553046988733]),
    coordinates([38.31513573999766, -76.54800508052972, 382.15340174455935]),
    coordinates([38.31513573999766, -76.54613407249012, 351.20312542538807]),
    coordinates([38.31513573999766, -76.54239205641093, 252.72501224443192]),
    coordinates([38.31571758935897, -76.5554891126881, 305.77411995952326]),
    coordinates([38.31571758935897, -76.5536181046485, 337.46067670527157]),
    coordinates([38.31571758935897, -76.5517470966089, 256.4968851671641]),
    coordinates([38.31571758935897, -76.5498760885693, 274.98608795923917]),
    coordinates([38.31571758935897, -76.54800508052972, 377.333131229825]),
    coordinates([38.31571758935897, -76.54613407249012, 307.95161713145467]),
    coordinates([38.316299438720286, -76.5554891126881, 338.6780896400912]),
    coordinates([38.316299438720286, -76.5536181046485, 391.13104403570173]),
    coordinates([38.316299438720286, -76.5517470966089, 363.5071356750003]),
    coordinates([38.316299438720286, -76.5498760885693, 231.49842064478904]),
    coordinates([38.316299438720286, -76.54800508052972, 380.0054724503371]),
    coordinates([38.316299438720286, -76.54613407249012, 341.5056942275929]),
    coordinates([38.316299438720286, -76.54426306445052, 253.08978380271313]),
    coordinates([38.316881288081596, -76.5554891126881, 367.6406099145077]),
    coordinates([38.316881288081596, -76.5517470966089, 350.8792752570124]),
    coordinates([38.316881288081596, -76.5498760885693, 223.73592556374786]),
    coordinates([38.316881288081596, -76.54800508052972, 373.24278466050805]),
    coordinates([38.316881288081596, -76.54613407249012, 349.12214935993813]),
    coordinates([38.316881288081596, -76.54426306445052, 399.4245790508111]),
    coordinates([38.317463137442914, -76.5517470966089, 220.397008464824]),
    coordinates([38.317463137442914, -76.5498760885693, 380.9504294065333]),
    coordinates([38.317463137442914, -76.54800508052972, 312.78641212652224]),
    coordinates([38.318044986804225, -76.5517470966089, 395.1111058859678]),
    coordinates([38.318044986804225, -76.5498760885693, 358.3101855940932])
    ])
'''