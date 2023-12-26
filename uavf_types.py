from typing import TypeAlias, TypeVar, Union, Any, Callable, Tuple, List
import jax.numpy as jnp
import jax
from sympy import symbols, Function
from dataclasses import dataclass
from mpmath import radians
import config
import utils
from jax.numpy.linalg import inv
from scipy.integrate import ode
from jax.numpy import sin, cos, tan, pi, sign
import matplotlib.pyplot as plt
import matplotlib
from ahrs import Quaternion


matplotlib.use('qtagg')

# Define types
# ---------------------------

coordinates : TypeAlias = list[float, float, float]
waypoints_global: TypeAlias = list[coordinates]
waypoints_local: TypeAlias = list[jnp.array]

class waypoints:

    flight_boundaries = [
        (38.31729702009844, -76.55617670782419),
        (38.31594832826572, -76.55657341657302),
        (38.31546739500083, -76.55376201277696),
        (38.31470980862425, -76.54936361414539),
        (38.31424154692598, -76.54662761646904),
        (38.31369801280048, -76.54342380058223),
        (38.31331079191371, -76.54109648475954),
        (38.31529941346197, -76.54052104837133),
        (38.31587643291039, -76.54361305817427),
        (38.31861642463319, -76.54538594175376),
        (38.31862683616554, -76.55206138505936),
        (38.31703471119464, -76.55244787859773),
        (38.31674255749409, -76.55294546866578),
        (38.31729702009844, -76.55617670782419)
    ]


    def __init__(self, global_origin : coordinates, waypoints : waypoints_global):
        self.global_origin = global_origin
        self.waypoints_global = waypoints

        self.waypoints_local = jnp.array([self.convert_to_ned(lat, lon, height) for lat, lon, height in self.waypoints_global])

    def convert_to_ned(self, lat, lon, alt):
        # Ensure the global_origin altitude is in feet MSL to match the altitude provided
        origin_lat, origin_lon, origin_alt = self.global_origin
        
        # Earth radius in meters
        R = 6369710  # Approximate radius of the Earth

        # Convert latitude and longitude from degrees to radians
        lat = jnp.radians(lat)
        lon = jnp.radians(lon)
        origin_lat = jnp.radians(origin_lat)
        origin_lon = jnp.radians(origin_lon)

        # Calculate the NED coordinates
        d_lat = lat - origin_lat
        d_lon = lon - origin_lon

        # Calculate the North and East components
        ned_north = R * d_lat
        ned_east = R * jnp.cos(origin_lat) * d_lon

        # Calculate the Down component (positive down)
        # Altitudes are converted from feet MSL to meters before the subtraction
        # to get the relative altitude in meters for the NED Down component
        ned_down = 1.0 * ((origin_alt * 0.3048) - (alt * 0.3048))

        return jnp.array([ned_north, ned_east, ned_down])
    
    def verbose(self):
        print("Global origin: {}".format(self.global_origin))
        print("Global waypoints: {}".format(self.waypoints_global))
        print("Local waypoints: {}".format(self.waypoints_local))

    def plot_waypoints(self):
        ax = plt.figure().add_subplot(projection='3d')
        
        # Convert flight boundaries to NED coordinates at ground level and at specified altitude
        boundary_ground_ned = jnp.array([self.convert_to_ned(lat, lon, 400) for lat, lon in self.flight_boundaries])
        boundary_altitude_ned = jnp.array([self.convert_to_ned(lat, lon, 142) for lat, lon in self.flight_boundaries])

        # Draw vertical lines (walls) for each boundary point
        for p1, p2 in zip(boundary_ground_ned, boundary_altitude_ned):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='lightblue')

        # Plot the boundary lines on top of the walls for clarity
        ax.plot(boundary_altitude_ned[:, 0], boundary_altitude_ned[:, 1], boundary_altitude_ned[:, 2], 'b-', label='Flight Boundaries')
        
        # Plot all waypoints in gray
        ax.scatter(self.waypoints_local[:, 0], self.waypoints_local[:, 1], self.waypoints_local[:, 2], color='gray', label='Waypoints')

        # Plot straight dotted lines between waypoints with increased opacity

        for i in range(len(self.waypoints_local) - 1):
            ax.plot(
                [self.waypoints_local[i, 0], self.waypoints_local[i+1, 0]],
                [self.waypoints_local[i, 1], self.waypoints_local[i+1, 1]],
                [self.waypoints_local[i, 2], self.waypoints_local[i+1, 2]],
                'k--',  # 'k--' represents a black dotted line
                alpha=0.7,  # Set the opacity here; 1.0 is fully opaque, 0.7 is less transparent
                label='Waypoint Order' if i == 0 else ""
            )
            
        # Plot the first waypoint in orange
        ax.scatter(self.waypoints_local[0, 0], self.waypoints_local[0, 1], self.waypoints_local[0, 2], color='orange', label='First Waypoint')

        # Plot the last waypoint in red
        ax.scatter(self.waypoints_local[-1, 0], self.waypoints_local[-1, 1], self.waypoints_local[-1, 2], color='red', label='Last Waypoint')

        # Plot the origin in green
        origin_ned = self.convert_to_ned(self.global_origin[0], self.global_origin[1], self.global_origin[2])
        ax.scatter(origin_ned[0], origin_ned[1], origin_ned[2], color='green', label='Origin')


        # Define the corners of the rectangular plane, assuming the flight boundaries provide the area.
        # Get the min and max values for North and East for the boundary corners.
        min_north, max_north = min(self.waypoints_local[:, 0]), max(self.waypoints_local[:, 0])
        min_east, max_east = min(self.waypoints_local[:, 1]), max(self.waypoints_local[:, 1])

        # Create a grid for the surface
        plane_north, plane_east = jnp.meshgrid(
            jnp.linspace(min_north, max_north, 2),
            jnp.linspace(min_east, max_east, 2)
        )
        plane_down = jnp.full(plane_north.shape, -22.86)  # Set the constant altitude for the plane

        # Plot the opaque red plane at the specified altitude
        ax.plot_surface(plane_north, plane_east, plane_down, color='red', alpha=0.5, label='Min Alt')  # Set the opacity with the alpha parameter
        
        # Set the limits and labels
        ax.set_zlim(-100, 0)
        ax.invert_zaxis()
        ax.set_xlabel('North X (m)')
        ax.set_ylabel('East Y (m)')
        ax.set_zlabel('Down Z (m)')

        # Include a legend
        ax.legend()
        
        plt.show()