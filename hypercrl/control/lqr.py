import scipy
import numpy as np
import torch

class LQR(object):
    def __init__(self, x_dim, a_dim, horizon):
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.T = horizon
        self.mass = 1
        self.dt = 0.1

    def set_env_gains(self, task_id):
        alpha = 0.5 * task_id
        m = self.mass
        dt = self.dt
        dx = self.x_dim
        da = self.a_dim

        self.A = np.eye(dx)
        self.A[:da, da:] = dt * np.eye(da)
        self.A[da:, da:] -= (alpha * dt / m) * np.eye(da)
        self.B = np.zeros((dx, da))
        self.B[da:, :] = dt/m * np.eye(da)
        self.Q = 0.1 * np.eye(dx)
        self.R = np.eye(da)

    def compute_policy_gains(self):
        T = self.T
        # Need to stabilize the system around error = 0, command = 0

        if type(self.A) != type([]):
            self.A = T*[self.A] 

        if type(self.B) != type([]):
            self.B = T*[self.B] 
            
        
        self.P = (T+1)*[self.Q]
        self.K = (T+1)*[0]
        
        for t in range(1, T + 1):
            
            self.K[t] = np.dot(self.B[T-t].transpose(), np.dot(self.P[t-1], self.A[T-t]))
            F = self.R + np.dot(self.B[T-t].transpose(), np.dot(self.P[t-1], self.B[T-t]))
            F = np.linalg.inv(F)

            self.K[t] = -np.dot(F, self.K[t])
            
            C = self.A[T-t] + np.dot(self.B[T-t], self.K[t])
            E = np.dot(self.K[t].transpose(), np.dot(self.R, self.K[t]))

            self.P[t] = self.Q + E + np.dot(C.transpose(), np.dot(self.P[t-1], C))

        
        self.K = self.K[1:]
        self.K = self.K[::-1]
        self.P = self.P[::-1]

        return self.K
    
    def command(self, state, task_id):
        self.set_env_gains(task_id)
        K = self.compute_policy_gains()

        state = state.reshape(-1, 1)
        u = np.dot(K[0], state)
        return torch.tensor(u)