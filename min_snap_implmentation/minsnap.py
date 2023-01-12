import numpy as np
import matplotlib.pyplot as plt
from math import factorial, atan2
from scipy.interpolate import PPoly

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver

import importlib

import pos_constraints
importlib.reload(pos_constraints)
from pos_constraints import Ab_i1

def add_pos_constraints(prog, sigma, n, d, w, dt):
  # Add A_i1 constraints here
  for i in range(n):
    Aeq_i, beq_i = Ab_i1(i, n, d, dt[i], w[i], w[i + 1])
    prog.AddLinearEqualityConstraint(Aeq_i, beq_i, sigma.flatten())

def add_continuity_constraints(prog, sigma, n, d, dt):
  # TDOO: Add A_i2 constraints here
  # Hint: Use AddLinearEqualityConstraint(expr, value)
  import sys
    

  for i in range(n-1):
      traj_sigma = sigma[i]
      traj_sigma_p1 = sigma[i+1]
      traj_sigma_y = traj_sigma[:, 0]
      traj_sigma_y_p1 = traj_sigma_p1[:, 0]
      traj_sigma_z = traj_sigma[:, 1]
      traj_sigma_z_p1 = traj_sigma_p1[:, 1]
      sigma_i_ip1 = sigma[i:(i+2)].reshape((2*sigma.shape[1], 2))
      sigma_y = sigma_i_ip1[:,0] #2 trajs contained
      sigma_z = sigma_i_ip1[:,1] #2 trajs contained

      
    #   print("sdvjldsvn")
    #   print(traj_sigma_y.shape)
      r1_arr = np.arange(0,d)
      A_1_y = np.zeros((4, 2*d)) #coz only y values in a single trajectory
      A_1_z = np.zeros((4, 2*d))
      B_1_y = np.zeros((4))
      B_1_z = np.zeros((4))

      A_1_y[0,:d] = r1_arr #k=1
      A_1_z[0,:d] = r1_arr
      A_1_y[0, d+1] = -1
      A_1_z[0, d+1] = -1

      #k=2
      k2_arr = [(kk-1)*(kk-2) for kk in range(1,d+1)]
    #   print(k2_arr)
    #   sys.exit()
      A_1_y[1,:d] = k2_arr
      A_1_y[1,d+2] = -2
      A_1_z[1,:d] = k2_arr
      A_1_z[1,d+2] = -2

      #k3
      k3_arr = [(kk-1)*(kk-2)*(kk-3) for kk in range(1,d+1)]
    #   print(len(k3_arr))
    #   sys.exit()
      A_1_y[2,:d] = k3_arr
      A_1_y[2,d+3] = -6
      A_1_z[2,:d] = k3_arr
      A_1_z[2,d+3] = -6

      #k4
      k4_arr = [(kk-1)*(kk-2)*(kk-3)*(kk-4) for kk in range(1,d+1)]
    #   print(k4_arr)
    #   print(len(k4_arr))
    #   sys.exit()
      A_1_y[3,:d] = k4_arr
      A_1_y[3,d+4] = -24
      A_1_z[3,:d] = k4_arr
      A_1_z[3,d+4] = -24


      prog.AddLinearEqualityConstraint(A_1_y, B_1_y, sigma_y.flatten())
      prog.AddLinearEqualityConstraint(A_1_z, B_1_z, sigma_z.flatten())
  
def add_minsnap_cost(prog, sigma, n, d, dt):
  # TODO: Add cost function here
  # Use AddQuadraticCost to add a quadratic cost expression
  import sys
  total_cost = 0
  for i in range(n):
    traj_sigma = sigma[i]
    # print(sigma.flatten()[2*i*d+8: 2*(i+1)*d :2])
    traj_sigma_y = traj_sigma[:, 0]
    traj_sigma_z = traj_sigma[:, 1]
    cy = 0
    cz = 0
    k = 0
    k2 = 0
    for j1 in range(4,d):
        for j2 in range(4,d):
            sigma_y1 = traj_sigma_y[j1]
            sigma_y2 = traj_sigma_y[j2]
            sigma_z1 = traj_sigma_z[j1]
            sigma_z2 = traj_sigma_z[j2]
            c1 = (j1)*(j1-1)*(j1-2)*(j1-3)
            c2 = (j2)*(j2-1)*(j2-2)*(j2-3)
            cy += (c1*c2*sigma_y1*sigma_y2*(dt[i]**(j1+j2-7)))/(j1+j2-7)
            cz += (c1*c2*sigma_z1*sigma_z2*(dt[i]**(j1+j2-7)))/(j1+j2-7)

    
    total_cost +=(cy+cz)
  prog.AddQuadraticCost(total_cost)




def minsnap(n, d, w, dt):
  n_dim = 2
  dim_names = ['y', 'z']

  prog = mp.MathematicalProgram()
  # sigma is a (n, n_dim, d) matrix of decision variables
  sigma = np.zeros((n, d, n_dim), dtype="object")
  for i in range(n):
    for j in range(d):
      sigma[i][j] = prog.NewContinuousVariables(n_dim, "sigma_" + str(i) + ',' +str(j)) 

  add_pos_constraints(prog, sigma, n, d, w, dt)
  add_continuity_constraints(prog, sigma, n, d, dt)
  add_minsnap_cost(prog, sigma, n, d, dt)  

  solver = OsqpSolver()
  result = solver.Solve(prog)
  print(result.get_solution_result())
  v = result.GetSolution()
  
  # Reconstruct the trajectory from the polynomial coefficients
  coeffs_y = v[::2]
  coeffs_z = v[1::2]
  y = np.reshape(coeffs_y, (d, n), order='F')
  z = np.reshape(coeffs_z, (d, n), order='F')
  coeff_matrix = np.stack((np.flip(y, 0), np.flip(z, 0)), axis=-1)  
  t0 = 0
  t = np.hstack((t0, np.cumsum(dt)))
  minsnap_trajectory = PPoly(coeff_matrix, t, extrapolate=False)

  return minsnap_trajectory

if __name__ == '__main__':

  n = 4
  d = 14

  w = np.zeros((n + 1, 2))
  dt = np.zeros(n)

  w[0] = np.array([-3,-4])
  w[1] = np.array([ 0, 0])
  w[2] = np.array([ 2, 3])
  w[3] = np.array([ 5, 0])
  w[4] = np.array([ 8, -2])

  dt[0] = 1
  dt[1] = 1
  dt[2] = 1
  dt[3] = 1

  # Target trajectory generation
  minsnap_trajectory = minsnap(n, d, w, dt)

  g = 9.81
  t0 = 0
  tf = sum(dt)
  n_points = 100
  t = np.linspace(t0, tf, n_points)

  fig = plt.figure(figsize=(4,3))
  ax = plt.axes()
  ax.scatter(w[:, 0], w[:, 1], c='r', label='way pts')
  ax.plot(minsnap_trajectory(t)[:,0], minsnap_trajectory(t)[:,1], label='min-snap trajectory')
  ax.set_xlabel("y")
  ax.set_ylabel("z")
  ax.legend()

  debugging = False
  # Set debugging to true to verify that the derivatives up to 5 are continuous
  if debugging:
    fig2 = plt.figure(figsize=(4,3))
    plt.plot(t, minsnap_trajectory(t,1)[:], label='1st derivative')
    plt.legend()

    fig3 = plt.figure(figsize=(4,3))
    plt.plot(t, minsnap_trajectory(t,2)[:], label='2nd derivative')
    plt.legend()

    fig4 = plt.figure(figsize=(4,3))
    plt.plot(t, minsnap_trajectory(t,3)[:], label='3rd derivative')
    plt.legend()

    fig5 = plt.figure(figsize=(4,3))
    plt.plot(t, minsnap_trajectory(t,4)[:], label='4th derivative')
    plt.legend()
    
  plt.show()  