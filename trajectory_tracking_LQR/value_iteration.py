import numpy as np
from math import *
from grid_world import *
import sys

import matplotlib.pyplot as plt

def plot_value_function_and_optimal_policy(world, V, u_opt):
  plt.clf()
  v_plot = plt.imshow(V, interpolation='nearest')
  colorbar = plt.colorbar()
  colorbar.set_label("Value function")
  plt.xlabel("Column")
  plt.ylabel("Row")
  arrow_length = 0.25
  for row in range(world.rows):
    for col in range(world.cols):
      if u_opt[row, col] == 0: #N
        plt.arrow(col, row, 0, -arrow_length, head_width=0.1)
      elif u_opt[row, col] == 1: #E
        plt.arrow(col, row, arrow_length, 0, head_width=0.1)
      elif u_opt[row, col] == 2: #S
        plt.arrow(col, row, 0, arrow_length, head_width=0.1)
      elif u_opt[row, col] == 3: #W
        plt.arrow(col, row, -arrow_length, 0, head_width=0.1)
      else:
        raise ValueError("Invalid action")
  plt.savefig('value_function.png', dpi=240)
  plt.show()

def value_iteration(world, threshold, gamma, plotting=True):
  V = np.zeros((world.rows, world.cols))
  V_old = np.zeros((world.rows, world.cols))
#   np.random.seed()
#   V = np.random.rand(world.rows, world.cols)
  u_opt = np.zeros((world.rows, world.cols))
  grid_x, grid_y = np.meshgrid(np.arange(0, world.rows, 1), 
                               np.arange(0, world.cols, 1))
  delta = 10.0

  fig = plt.figure("Gridworld")
  
  # STUDENT CODE: calculate V and the optimal policy u using value iteration
  g = world
  p = g.P
#   while(1):
#     for i in range(world.rows):
#         for j in range(world.cols):
            
#             s = g.map_row_col_to_state(i,j)
#             # print(p[s])
#             val=np.array([np.array(xi) for xi in p[s]])
#             # print("val")
#             # print(val)
#             probs = val[:,:,0]
#             costs = val[:,:,2]
#             next_states = val[:,:,1].astype(int)
#             try_v_0 = []
#             try_v_1 = []
#             for o in next_states[:,0]:
#                 indx_0 = V[g.map_state_to_row_col(o)]
#                 try_v_0.append(indx_0)

#             for jj in next_states[:,1]:
#                 indx_1 = V[g.map_state_to_row_col(jj)]
#                 try_v_1.append(indx_1)
            
#             # sys.exit()

#             # print(try_v)
#             val_array_0 = np.array(try_v_0)*gamma
#             val_array_1 = np.array(try_v_1)*gamma
#             # print("bjd")
#             # indx_0 = g.map_state_to_row_col(next_states[:,0])
#             cplusv = np.zeros((costs.shape[0], costs.shape[1]))
#             cplusv[:,0] = costs[:,0] + val_array_0 #no slip 
#             cplusv[:,1] = costs[:,1] + val_array_1 #slip
#             exp = probs*(cplusv)
#             exp_val = exp[:,0]+exp[:,1]
#             min_action = np.argmin(exp_val)
#             min_val = np.min(exp_val)      
#             V[i][j] = min_val
#             u_opt[i][j] = min_action
#     # print(V)
#     if(np.any(np.abs(V - V_old)) < threshold):
#         # print((V)[0])
#         # print("sdvmnjfsbnvjdsvbjldsvl")
#         break
#     else:
#         V_old = V
  iter_ = 0
  while(1):
    for i in range(world.rows):
        for j in range(world.cols):   
            s = g.map_row_col_to_state(i,j)
            # print(p[s])
            val=np.array([np.array(xi) for xi in p[s]])
            pot_vals = []
            for ac in val:
                # print("ac")
                # print(ac)
                exp = 0
                for oneac in ac:
                    val = V[g.map_state_to_row_col(oneac[1].astype(int))]
                    cost = oneac[2]
                    prob = oneac[0]
                    exp += prob*(cost+ gamma*val)
                    # print(exp)
                pot_vals.append(exp)
            # print(pot_vals)
            # sys.exit()
            min_action = np.argmin(pot_vals)
            min_val = np.min(pot_vals)      
            V[i][j] = min_val
            u_opt[i][j] = min_action
    # print(V)
    # if(np.any(np.abs(V - V_old)) < threshold):
    #     # print((V)[0])
    #     # print("sdvmnjfsbnvjdsvbjldsvl")
    #     break
    # else:
    #     V_old = V
    break_flag = True
    comp = np.abs(V_old - V)
    iter_+=1

    for nn in comp.flatten():
        if(nn > threshold):
            break_flag = False
    if(break_flag== False):
        V_old = V.copy()
    else:
        print(iter_)
        break
                

  if plotting:
    plot_value_function_and_optimal_policy(world, V, u_opt)

  return V, u_opt

if __name__=="__main__":
  world = Gridworld() 
  threshold = 0.0001
  gamma = 0.9
  # value_iteration(world, threshold, gamma, False)
  value_iteration(world, threshold, gamma, True)
