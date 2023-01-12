import numpy as np
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp

from trajectories import *
import sys
class Quadrotor(object):
  '''
  Constructor. Compute function S(t) using S(t) = L(t) L(t)^t, by integrating backwards
  from S(tf) = Qf. We will then use S(t) to compute the optimal controller efforts in 
  the compute_feedback() function
  '''
  def __init__(self, Q, R, Qf, tf):
    self.m = 1
    self.a = 0.25
    self.I = 0.0625
    self.Q = Q
    self.R = R
    self.B = np.zeros((6,2))

    ''' 
    We are integrating backwards from Qf
    '''

    # Get L(tf) L(tf).T = S(tf) by decomposing S(tf) using Cholesky decomposition
    L0 = cholesky(Qf).transpose()

    # We need to reshape L0 from a square matrix into a row vector to pass into solve_ivp()
    l0 = np.reshape(L0, (36))
    # L must be integrated backwards, so we integrate L(tf - t) from 0 to tf
    initial_condition = [0, tf]
    sol = solve_ivp(self.dldt_minus, [0, tf], l0, dense_output=True)
    t = sol.t
    l = sol.y
    # print(t)
    # print(l)

    # Reverse time to get L(t) back in forwards time
    t = tf - t
    t = np.flip(t)
    l = np.flip(l, axis=1) # flip in time
    # print(l.shape)
    self.l_spline = interp1d(t, l)

  def Ldot(self, t, L):

    x = x_d(t)
    u = u_d(t)
    Q = self.Q
    R = self.R

    dLdt = np.zeros((6,6))
    # STUDENT CODE: compute d/dt L(t)
    theta_d = x[2]

    u_1 = u[0]
    u_2 = u[1]
    At = np.array([[0,0,0,1,0,0],
                   [0,0,0,0,1,0],
                   [0,0,0,0,0,1],
                   [0,0,-(np.cos(theta_d)*(u_1+u_2))/self.m,0,0,0],
                   [0,0,-(np.sin(theta_d)*(u_1+u_2))/self.m,0,0,0],
                   [0,0,0,0,0,0]])

    Bt = np.array([[0,0],
                   [0,0],
                   [0,0],
                   [-np.sin(theta_d)/self.m,-np.sin(theta_d)/self.m],
                   [np.cos(theta_d)/self.m,np.cos(theta_d)/self.m],
                   [self.a/self.I, -self.a/self.I]])
    
    self.B = Bt
    dLdt = -0.5*(self.Q @ np.linalg.inv(L.T)) - (At.T @ L) + (0.5*L @ L.T @ Bt @ np.linalg.inv(self.R) @ Bt.T @ L)
    # print(At.shape)
    # print(Bt.shape)

    # sys.exit()

    return dLdt

  def dldt_minus(self, t, l):
    # reshape l to a square matrix
    L = np.reshape(l, (6, 6))

    # compute Ldot
    dLdt_minus = -self.Ldot(t, L)

    # reshape back into a vector
    dldt_minus = np.reshape(dLdt_minus, (36))
    return dldt_minus


  def compute_feedback(self, t, x):
    theta_d = x_d(t)[2]
    Bt = np.array([[0,0],
                   [0,0],
                   [0,0],
                   [-np.sin(theta_d)/self.m,-np.sin(theta_d)/self.m],
                   [np.cos(theta_d)/self.m,np.cos(theta_d)/self.m],
                   [self.a/self.I, -self.a/self.I]])

    # Retrieve L(t)
    L = np.reshape(self.l_spline(t), (6, 6))

    u_fb = np.zeros((2,))
    # STUDENT CODE: Compute optimal feedback inputs u_fb using LQR
    St = L@ L.T
    u_fb = -np.linalg.inv(self.R)@ Bt.T@ St @ (x - x_d(t))
    # Add u_fb to u_d(t), the feedforward term. 
    # u = u_fb + u_d
    u = u_d(t) + u_fb;
    return u