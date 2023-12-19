from uavf_types import cpc_trajectory_optimization, coordinates, waypoints_global
import jax.numpy as jnp

def main():

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

if __name__ == "__main__":
    main()