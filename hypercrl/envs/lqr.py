import numpy as np
import torch
import matplotlib.pyplot as plt

import gym

class LQR():
    x_dim = None
    u_dim = None

    def __init__(self, noise):
        self.noise = noise
        self.A, self.B, self.Q, self.R = [None]*4
        self.states = []
        self.time_step = 0

    def close(self):
        pass

    def reset(self):
        self.time_step = 0
        self.x_t = np.random.randn(self.x_dim, 1)
        self.states.append(self.x_t)
        return self.x_t.copy()
    
    def terminate(self, x):
        raise NotImplementedError
   
    def cost(self, x, u):
        cost = x.T @ self.Q @ x + u.T @ self.R @ u
        return float(cost)
    
    def reward_torch(self, x, u):
        """
        x : state tensor (size [particle, Batch, horizon, dim])
        u: action tensor (size [Batch, horizon, dim])
        """
        Q = torch.tensor(self.Q, device=x.device, dtype=x.dtype)
        R = torch.tensor(self.R, device=x.device, dtype=x.dtype)

        x_cost = x.unsqueeze(-2) @ Q @ x.unsqueeze(-1)
        u_cost = u.unsqueeze(-2) @ R @ u.unsqueeze(-1)

        # Remove extra dimension from batch matrix multiplication
        x_cost.squeeze_(-1).squeeze_(-1)
        u_cost.squeeze_()

        cost = x_cost + u_cost
        return - cost

    def step(self, u_t): 
        """
        param u_t: action
        return:
            x_tt: next state
            cost: -reward
            done: 
            info: 
        """
        if isinstance(u_t, torch.Tensor):
            u_t = u_t.detach().cpu().numpy()
        self.time_step += 1
        x_t = self.x_t
        
        w_t = np.random.randn(self.x_dim, 1) * self.noise

        x_tt = self.A @ x_t + self.B @ u_t + w_t
        cost = self.cost(x_t, u_t)
        done = self.terminate(x_tt)
        info = {}
        
        self.x_t = x_tt.copy()
        self.states.append(self.x_t)
     
        return x_tt, -cost, done, info


class LQR_2DCar(LQR):
    x_dim = 4
    u_dim = 2
    def __init__(self, noise=0, friction=0, mass=1):
        super(LQR_2DCar, self).__init__(noise)
        self.dt = 0.1
        self.bound = 10
        self.center = np.array([[0.], [0.]])

        self.set_param(friction, mass)
        self.action_space = gym.spaces.Box(-10, 10, shape=(2, 1))
        self.Q = 0.1 * np.eye(4)
        self.R = np.eye(2)

        self.fig = None
    
    def set_param(self, friction=0, mass=1):
        alpha = friction
        m = mass
        dt = self.dt
        self.A = np.array([[1, 0, dt,                  0],
                           [0, 1, 0,                   dt],
                           [0, 0, 1 - alpha * dt / m,  0],
                           [0, 0, 0,                   1 - alpha * dt / m]])
        self.B = np.array([[0,      0],
                           [0,      0],
                           [dt / m, 0],
                           [0, dt / m]])

    def reset(self, x=None):
        self.time_step = 0
        if x is not None:
            self.x_t = x.copy()
        else:
            self.x_t = np.random.randn(self.x_dim, 1)
            self.x_t[0:2] = self.x_t[0:2] * 2.5 + self.center
        return self.x_t.copy()
    
    def terminate(self, x):
        norm = np.linalg.norm(x[:2])
        vel_norm = np.linalg.norm(x[2:])
        
        if (abs(x[0, 0]) > self.bound) or (abs(x[1, 0]) > self.bound):
            return True

        if (norm <= 1e-4) and (vel_norm <= 1e-4):
            return True
        else:
            return False
    
    def render(self):
        if self.fig is None:
            plt.ion()
            f, ax = plt.subplots(figsize=(5, 5))
            f.show()
            f.canvas.draw()
            self.fig = (f, ax)
        else:
            f, ax = self.fig
        
        ax.set_xlim(-self.bound, self.bound)
        ax.set_ylim(-self.bound, self.bound)

        x = np.hstack(self.states)
        ax.plot(x[0, :], x[1, :], 'kx')
        f.canvas.draw()

class LQR_HARD(LQR):
    x_dim = 20
    u_dim = 10
    episode_length = 400
    def __init__(self, noise=0, friction=0, mass=1):
        super(LQR_HARD, self).__init__(noise)
        self.dt = 0.1
        self.bound = 10
        self.center = np.zeros((10, 1))

        self.set_param(friction, mass)
        self.action_space = gym.spaces.Box(-10, 10, shape=(10, 1))
        self.Q = 0.1 * np.eye(20)
        self.R = np.eye(10)

        self.fig = None

    def set_param(self, friction=0, mass=1):
        alpha = friction
        m = mass
        dt = self.dt
        self.A = np.eye(20)
        self.A[:10, 10:] = dt * np.eye(10)
        self.A[10:, 10:] -= (alpha * dt / m) * np.eye(10)
        self.B = np.zeros((20, 10))
        self.B[10:, :] = dt/m * np.eye(10)

    def reset(self, x=None):
        self.time_step = 0
        if x is not None:
            self.x_t = x.copy()
        else:
            self.x_t = np.random.randn(self.x_dim, 1)
            self.x_t[0:10] = self.x_t[0:10] * 2.5 + self.center
        return self.x_t.copy()
    
    def terminate(self, x):
        if self.time_step >= self.episode_length:
            return True
    
        norm = np.linalg.norm(x[:10])
        vel_norm = np.linalg.norm(x[10:])
        
        abs_x = np.abs(x[:10, 0])
        if np.max(abs_x) > self.bound:
            return True

        if (norm <= 1e-4) and (vel_norm <= 1e-4):
            return True
        else:
            return False
