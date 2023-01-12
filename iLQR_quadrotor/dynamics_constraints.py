import numpy as np

from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.autodiffutils import AutoDiffXd


def EvaluateDynamics(planar_arm, context, x, u):
  # Computes the dynamics xdot = f(x,u)

  planar_arm.SetPositionsAndVelocities(context, x)
  n_v = planar_arm.num_velocities()

  M = planar_arm.CalcMassMatrixViaInverseDynamics(context)
  B = planar_arm.MakeActuationMatrix()
  g = planar_arm.CalcGravityGeneralizedForces(context)
  C = planar_arm.CalcBiasTerm(context)

  M_inv = np.zeros((n_v,n_v)) 
  if(x.dtype == AutoDiffXd):
    M_inv = AutoDiffXd.inv(M)
  else:
    M_inv = np.linalg.inv(M)
  v_dot = M_inv @ (B @ u + g - C)
  return np.hstack((x[-n_v:], v_dot))

def CollocationConstraintEvaluator(planar_arm, context, dt, x_i, u_i, x_ip1, u_ip1):
  n_x = planar_arm.num_positions() + planar_arm.num_velocities()
  h_i = np.zeros(n_x,)
  # TODO: Add a dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt
  # You should make use of the EvaluateDynamics() function to compute f(x,u)
  h_i = 3.0*(x_ip1 - x_i)/(2.0*dt) - (EvaluateDynamics(planar_arm, context, x_i, u_i)+ EvaluateDynamics(planar_arm, context, x_ip1, u_ip1))/4.0 - EvaluateDynamics(planar_arm, context, (((x_i+x_ip1)/2) - (dt/8.0)*(EvaluateDynamics(planar_arm, context, x_ip1, u_ip1)- EvaluateDynamics(planar_arm, context, x_i, u_i))), (u_i+u_ip1)/2)
#   print(h_i)
  return h_i

def AddCollocationConstraints(prog, planar_arm, context, N, x, u, timesteps):
  n_u = planar_arm.num_actuators()
  n_x = planar_arm.num_positions() + planar_arm.num_velocities()
  
  for i in range(N - 1):
    def CollocationConstraintHelper(vars):
      x_i = vars[:n_x]
      u_i = vars[n_x:n_x + n_u]
      x_ip1 = vars[n_x + n_u: 2*n_x + n_u]
      u_ip1 = vars[-n_u:]
      print(CollocationConstraintEvaluator(planar_arm, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1))
      return CollocationConstraintEvaluator(planar_arm, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1)
      
    # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
    #       to prog
    # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
    # where vars = hstack(x[i], u[i], ...)
    ub = np.zeros((n_x))
    lb = np.zeros((n_x))
    vars_array = np.concatenate((x[i], u[i]))
    vars_array = np.concatenate((vars_array, x[i+1]))
    vars_array = np.concatenate((vars_array, u[i+1]))

    # vars_array = np.concatenate(np.concatenate((x[i], u[i])), (x[i+1],u[i+1]))
    prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars_array)