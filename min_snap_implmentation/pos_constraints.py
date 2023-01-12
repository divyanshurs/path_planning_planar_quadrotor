import numpy as np
import sys
def Ab_i1(i, n, d, dt_i, w_i, w_ip1):
  '''
  Ab_i1(i, n, d, dt_i, w_i, w_ip1) computes the linear equality constraint
  constants that require the ith polynomial to meet waypoints w_i and w_{i+1}
  at it's endpoints.
  Parameters:
     i - index of the polynomial.
     n - total number of polynomials.
     d - the number of terms in each polynomial.
     dt_i - Delta t_i, duration of the ith polynomial.
     w_i - waypoint at the start of the ith polynomial.
     w_ip1 - w_{i+1}, waypoint at the end of the ith polynomial.
  Outputs
     A_i1 - A matrix from linear equality constraint A_i1 v = b_i1
     b_i1 - b vector from linear equality constraint A_i1 v = b_i1
  '''

  A_i1 = np.zeros((4, 2*d*n))
  b_i1 = np.zeros((4, 1))

  # TODO: fill in values for A_i1 and b_i1
  r1=[]
  r2 = []
  r3 = []
  r4 = []
  r3_track = 1
  r4_track = 1
#   print(n)

#   sys.exit()
  for p in range(2*d):
    #   t_arr.append(np.power(dt_i,p))
    if(p ==0):
        r3.append(1)
        r4.append(0)
    if(p==1):
        r3.append(0)
        r4.append(1)
    if(p>1):
        if(p%2==0):
            r3.append(np.power(dt_i,r3_track))
            r3_track+=1
            r4.append(0)
        else:
            r3.append(0)
            r4.append(np.power(dt_i,r4_track))
            r4_track+=1

  A_i1[0, (2*i*d)] = 1
  A_i1[1, 2*i*d+1] = 1
  A_i1[2, (2*i*d):(2*i*d + 2*d)] = r3
  A_i1[3, (2*i*d):(2*i*d + 2*d)] = r4

#   t_arr = np.array(t_arr)
    # print(w_i.shape)
  b_i1[:2] = w_i.reshape((2,-1))
  b_i1[2:] = w_ip1.reshape((2,-1))
#   A_i1[1,1:] = t_arr[0]
  return A_i1, b_i1