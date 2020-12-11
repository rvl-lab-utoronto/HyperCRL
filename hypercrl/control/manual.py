import scipy
import numpy as np
import torch

class Manual(object):
    def __init__(self, env, x_dim, a_dim, horizon, dynamics, gpuid):
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.env = env
        self.horizon = horizon
        self.dynamics = dynamics
        self.gpuid = gpuid

    def plan(self, x, task_id):
        x = x.view(self.x_dim)
        if self.env == "pusher":
            goal = 0.06
            if x[2] > goal:
                action = torch.tensor([0., 0.], device=self.gpuid)
            else:
                action = torch.randn(2, device=self.gpuid) * torch.tensor([0.3, 0.3], device=self.gpuid) \
                        + torch.tensor([1.0, 0.], device=self.gpuid)
            return action

    def command(self, state, task_id, first_action=True):
        # Convert to torch
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)

        if first_action:
            return self.plan(state, task_id)

        actions = []
        for t in range(self.horizon):
            action = self.plan(state, task_id)
            state = self.dynamics(state, action, task_id)
            actions.append(action)

        return actions

    def reset(self):
        pass