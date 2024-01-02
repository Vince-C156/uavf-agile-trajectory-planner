from control import FiniteTimeLQR
from dynamics import CompQuad_
import numpy as np
from uavf_types import waypoints, coordinates, waypoints_global
import matplotlib.pyplot as plt
from pydrake.all import (AutoDiffXd,FirstOrderTaylorApproximation, DirectCollocation, DiagramBuilder, LeafSystem, Solve, MathematicalProgram, SnoptSolver, Linearize, Expression)

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
x = np.array([0.0, 0.0, -1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])

mission_waypoints = waypoints(global_origin = global_origin, waypoints = wps)
plant = CompQuad_[None]()

myLQR = FiniteTimeLQR(plant, x, 16, mission_waypoints, 100)