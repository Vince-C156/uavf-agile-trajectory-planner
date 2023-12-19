"""
Imports
"""
# The simulator is instantiated using the Environment class
from rotorpy.environments import Environment
from uavf_types import cpc_trajectory_optimization, coordinates, waypoints_global
import jax.numpy as jnp
# Vehicles. Currently there is only one. 
# There must also be a corresponding parameter file. 
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
# from rotorpy.vehicles.hummingbird_params import quad_params  # There's also the Hummingbird

# You will also need a controller (currently there is only one) that works for your vehicle. 
from rotorpy.controllers.quadrotor_control import SE3Control

# And a trajectory generator
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap

# You can optionally specify a wind generator, although if no wind is specified it will default to NoWind().
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind
from rotorpy.wind.dryden_winds import DrydenGust, DrydenGustLP
from rotorpy.wind.spatial_winds import WindTunnel

# You can also optionally customize the IMU and motion capture sensor models. If not specified, the default parameters will be used. 
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture

# You can also specify a state estimator. This is optional. If no state estimator is supplied it will default to null. 
from rotorpy.estimators.wind_ukf import WindUKF

# Also, worlds are how we construct obstacles. The following class contains methods related to constructing these maps. 
from rotorpy.world import World

# Reference the files above for more documentation. 

# Other useful imports
import numpy as np                  # For array creation/manipulation
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc. 
import os                           # For path generation

"""
Instantiation
"""

# Obstacle maps can be loaded in from a JSON file using the World.from_file(path) method. Here we are loading in from 
# an existing file under the rotorpy/worlds/ directory. However, you can create your own world by following the template
# provided (see rotorpy/worlds/README.md), and load that file anywhere using the appropriate path.
world = World.from_file(os.path.join(os.path.dirname(__file__), "worlds", "empty.json"))
# Global origin
global_origin = coordinates([38.31510824923089, -76.54914848803199])

# Waypoints
wps = waypoints_global([
    coordinates([38.31575393717897, -76.55448976093469]),
    coordinates([38.31680124885777, -76.55379223985771]),
    coordinates([38.31625280442319, -76.55051958493638]),
    coordinates([38.31662674426038, -76.54845431726757]),
    coordinates([38.31465730611633, -76.54522932236937]),
])

# Initial state
x = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

traj_optimizer = cpc_trajectory_optimization(global_origin = global_origin, waypoints = wps, x0 = x, dt = 1, T = 300)
# "world" is an optional argument. If you don't load a world it'll just provide an empty playground! 

# An instance of the simulator can be generated as follows: 
sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified. 
                           controller=SE3Control(quad_params),        # controller object, must be specified.
                           trajectory=CircularTraj(radius=2),         # trajectory object, must be specified.
                           wind_profile=SinusoidWind(),               # OPTIONAL: wind profile object, if none is supplied it will choose no wind. 
                           sim_rate     = 100,                        # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                           imu          = None,                       # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                           mocap        = None,                       # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.  
                           estimator    = None,                       # OPTIONAL: estimator object
                           world        = world,                      # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                           safety_margin= 0.25                        # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                       )

# This generates an Environment object that has a unique vehicle, controller, and trajectory.  
# You can also optionally specify a wind profile, IMU object, motion capture sensor, estimator, 
# and the simulation rate for the simulator.  

"""
Execution
"""

# Setting an initial state. This is optional, and the state representation depends on the vehicle used. 
# Generally, vehicle objects should have an "initial_state" attribute. 
x0 = {'x': np.array([0,0,0]),
      'v': np.zeros(3,),
      'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
      'w': np.zeros(3,),
      'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
      'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
sim_instance.vehicle.initial_state = x0

# Executing the simulator as specified above is easy using the "run" method: 
# All the arguments are listed below with their descriptions. 
# You can save the animation (if animating) using the fname argument. Default is None which won't save it.

results = sim_instance.run(t_final      = 20,       # The maximum duration of the environment in seconds
                           use_mocap    = False,       # Boolean: determines if the controller should use the motion capture estimates. 
                           terminate    = False,       # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                           plot            = True,     # Boolean: plots the vehicle states and commands   
                           plot_mocap      = True,     # Boolean: plots the motion capture pose and twist measurements
                           plot_estimator  = True,     # Boolean: plots the estimator filter states and covariance diagonal elements
                           plot_imu        = True,     # Boolean: plots the IMU measurements
                           animate_bool    = True,     # Boolean: determines if the animation of vehicle state will play. 
                           animate_wind    = True,    # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV. 
                           verbose         = True,     # Boolean: will print statistics regarding the simulation. 
                           fname   = None # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/. 
                    )

# There are booleans for if you want to plot all/some of the results, animate the multirotor, and 
# if you want the simulator to output the EXIT status (end time reached, out of control, etc.)
# The results are a dictionary containing the relevant state, input, and measurements vs time.

# To save this data as a .csv file, you can use the environment's built in save method. You must provide a filename. 
# The save location is rotorpy/data_out/
sim_instance.save_to_csv("basic_usage.csv")

# Instead of producing a CSV, you can manually unpack the dictionary into a Pandas DataFrame using the following: 
from rotorpy.utils.postprocessing import unpack_sim_data
dataframe = unpack_sim_data(results)