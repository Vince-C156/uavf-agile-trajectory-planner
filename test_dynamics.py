from pydrake.all import (AutoDiffXd, DirectCollocation, DiagramBuilder,
                         FirstOrderTaylorApproximation, PiecewisePolynomial, Solve)
import numpy as np
from dynamics import CompQuad_
import matplotlib.pyplot as plt
# Instantiate your CompQuad system (make sure this is defined correctly)
CompQuad = CompQuad_[None]

plant = CompQuad()
context = plant.CreateDefaultContext()

x0 = np.array([0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Combined state for hovering.
xf = np.array([10, 10, -10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Combined state for hovering.

context.SetContinuousState(x0)
dircol = DirectCollocation(
    plant,
    context,
    num_time_samples=50,
    minimum_time_step=0.1,
    maximum_time_step=1,
    input_port_index=plant.get_input_port().get_index())

prog = dircol.prog() 
dircol.AddEqualTimeIntervalsConstraints()


prog.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())
prog.AddBoundingBoxConstraint(xf, xf, dircol.final_state())

dircol.AddFinalCost(dircol.time() ** 2)


result = Solve(prog)

print(result.is_success())

x_sol = dircol.ReconstructStateTrajectory(result)
u_sol = dircol.ReconstructInputTrajectory(result)

fig, ax = plt.subplots(3, 1)
fig.tight_layout(pad=2.0)
x_values = x_sol.vector_values(x_sol.get_segment_times())
ax[0].plot(range(len(x_values[0])), x_values[0, :], ".-")
ax[0].set_xlabel("t")
ax[0].set_ylabel("x")

ax[1].plot(range(len(x_values[1])), x_values[1, :], ".-")
ax[1].set_xlabel("t")
ax[1].set_ylabel("y")

ax[2].plot(range(len(x_values[2])), x_values[2, :], ".-")
ax[2].set_xlabel("t")
ax[2].set_ylabel("z")

plt.show()

fig, ax = plt.subplots(4, 1)

u_values = u_sol.vector_values(u_sol.get_segment_times())
ax[0].plot(range(len(u_values[0])), u_values[0, :], ".-")
ax[0].set_xlabel("t")
ax[0].set_ylabel("u1")

ax[1].plot(range(len(u_values[1])), u_values[1, :], ".-")
ax[1].set_xlabel("t")
ax[1].set_ylabel("u2")

ax[2].plot(range(len(u_values[2])), u_values[2, :], ".-")
ax[2].set_xlabel("t")
ax[2].set_ylabel("u3")

ax[3].plot(range(len(u_values[3])), u_values[3, :], ".-")
ax[3].set_xlabel("t")
ax[3].set_ylabel("u4")

plt.show()