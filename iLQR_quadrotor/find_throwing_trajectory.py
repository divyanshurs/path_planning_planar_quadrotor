import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)

import kinematic_constraints
import dynamics_constraints
importlib.reload(kinematic_constraints)
importlib.reload(dynamics_constraints)
from kinematic_constraints import (
  AddFinalLandingPositionConstraint
)
from dynamics_constraints import (
  AddCollocationConstraints,
  EvaluateDynamics
)
import sys

def find_throwing_trajectory(N, initial_state, final_configuration, distance, tf):
  '''
  Parameters:
    N - number of knot points
    initial_state - starting configuration
    distance - target distance to throw the ball

  '''

  builder = DiagramBuilder()
  plant = builder.AddSystem(MultibodyPlant(0.0))
  file_name = "planar_arm.urdf"
  Parser(plant=plant).AddModelFromFile(file_name)
  plant.Finalize()
  planar_arm = plant.ToAutoDiffXd()

  plant_context = plant.CreateDefaultContext()
  context = planar_arm.CreateDefaultContext()

  # Dimensions specific to the planar_arm
  n_q = planar_arm.num_positions()
  n_v = planar_arm.num_velocities()
  n_x = n_q + n_v
  n_u = planar_arm.num_actuators()

  # Store the actuator limits here
  effort_limits = np.zeros(n_u)
  for act_idx in range(n_u):
    effort_limits[act_idx] = \
      planar_arm.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit()
  joint_limits = np.pi * np.ones(n_q)
  vel_limits = 15 * np.ones(n_v)

  # Create the mathematical program
  prog = MathematicalProgram()
  x = np.zeros((N, n_x), dtype="object")
  u = np.zeros((N, n_u), dtype="object")
  for i in range(N):
    x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
    u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

  t_land = prog.NewContinuousVariables(1, "t_land")

  t0 = 0.0
  timesteps = np.linspace(t0, tf, N)
  x0 = x[0]
  xf = x[-1]

  # DO NOT MODIFY THE LINES ABOVE

  # Add the kinematic constraints (initial state, final state)
  # TODO: Add constraints on the initial state
  prog.AddLinearEqualityConstraint(x0, initial_state)
  
  # Add the kinematic constraint on the final state
  AddFinalLandingPositionConstraint(prog, xf, distance, t_land)

  # Add the collocation aka dynamics constraints
  AddCollocationConstraints(prog, planar_arm, context, N, x, u, timesteps)

  # TODO: Add the cost function here
  print("iCAME BACK")
  cost = 0
  for j in range(N-1):
    #   print("fb.")
    delta_t = timesteps[j+1] - timesteps[j]
    cost += (delta_t/2)*(u[j].T@u[j] + u[j+1].T@u[j+1])
    # print(u[j].T@u[j])

  prog.AddQuadraticCost(cost)

#   for i in range(N-2):
#     dt= timesteps[i+1]-timesteps[i]
#     Q   = dt*np.eye(4)
#     b   = np.zeros((4,1))
#     vars_arr= np.array([u[i],u[i+1]]).reshape((-1,1))
#     prog.AddQuadraticCost(Q,b,vars_arr,is_convex=True)


  # TODO: Add bounding box constraints on the inputs and qdot 
  import sys
  b_arr = np.zeros_like(x[:,:2])
  b_arr[:,0] = joint_limits[0]
  b_arr[:,1] = joint_limits[1]
  bv_arr = np.zeros_like(x[:,2:])
  bv_arr[:,0] = vel_limits[0]
  bv_arr[:,1] = vel_limits[1]

  t_arr = np.zeros_like(u[:,0])
  t2_arr = np.zeros_like(u[:,1])
  t_arr = effort_limits[0]
  t2_arr = effort_limits[1]

  prog.AddBoundingBoxConstraint(-b_arr, b_arr, x[:,:2])
  prog.AddBoundingBoxConstraint(-bv_arr, bv_arr, x[:,2:])
  prog.AddBoundingBoxConstraint(-t_arr, t_arr, u[:,0])
  prog.AddBoundingBoxConstraint(-t2_arr, t2_arr, u[:,1])


  # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)
  g_arr = np.zeros_like(x[:,:2])
  spaced_jlim = np.linspace(-joint_limits[0], joint_limits[0], g_arr.shape[0])
  g_arr[:,0] = spaced_jlim
  g_arr[:,1] = spaced_jlim

  
  ug_arr = np.zeros_like(u)
  inc_t = np.linspace(-effort_limits[0], effort_limits[0], ug_arr[:,0].shape[0])
  ug_arr[:,0] = inc_t
  ug_arr[:,1] = inc_t

  prog.SetInitialGuess(x[:,:2], g_arr)
  prog.SetInitialGuess(u, ug_arr)


  #DO NOT MODIFY THE LINES BELOW

  # Set up solver
  result = Solve(prog)
  
  x_sol = result.GetSolution(x)
  u_sol = result.GetSolution(u)
  t_land_sol = result.GetSolution(t_land)

  print('optimal cost: ', result.get_optimal_cost())
  print('x_sol: ', x_sol)
  print('u_sol: ', u_sol)
  print('t_land: ', t_land_sol)

  print(result.get_solution_result())

  # Reconstruct the trajectory
  xdot_sol = np.zeros(x_sol.shape)
  for i in range(N):
    xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])
  
  x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
  u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

  return x_traj, u_traj, prog, prog.GetInitialGuess(x), prog.GetInitialGuess(u)
  
if __name__ == '__main__':
  N = 5
  initial_state = np.zeros(4)
  final_configuration = np.array([np.pi, 0])
  tf = 3.0
  distance = 15.0
  x_traj, u_traj, prog, _, _ = find_throwing_trajectory(N, initial_state, final_configuration, distance, tf)