import torch
from torch import jit
from torch import nn, optim


class GradPlan():  # jit.ScriptModule):
    def __init__(self, dynamics, cost, nx, nu, samples, opt_iters, planning_horizon,
            device, dtype=torch.float32, grad_clip=True, init_covar_diag=1):
        super().__init__()
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K = samples
        self.device = device
        self.dtype = dtype
        self.grad_clip = grad_clip
        self.a_size = nu
        self._dynamics = dynamics
        self._cost = cost
        self.init_covar_diag = init_covar_diag

    def reset(self):
        pass

    def evaluate(self, init_states, actions, task_id):
        N = actions.size(1)
        states = init_states.view(1, -1).repeat(N, 1)
        cost_total = 0
        for t in range(self.H):
            u = actions[t]
            states = self._dynamics(states, u, task_id)
            cost_total += self._cost(states, u, t, task_id)
        return cost_total

    @torch.enable_grad()
    def command(self, state, task_id, first_action=True, return_plan_each_iter=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        state = state.to(dtype=self.dtype, device=self.device)
        B = 1

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = torch.zeros(self.H, B, 1, self.a_size, device=self.device)
        a_std = torch.ones(self.H, B, 1, self.a_size, device=self.device)

        # Sample actions (T x (B*K) x A)
        actions = (a_mu + a_std * torch.randn(self.H, B, self.K, self.a_size, device=self.device)).view(self.H, B * self.K, self.a_size)
        # TODO: debug
        # actions = actions*0
        actions = actions.clone().detach().requires_grad_(True)

        # optimizer = optim.SGD([actions], lr=0.1, momentum=0)
        optimizer = optim.RMSprop([actions], lr=0.1)
        plan_each_iter = []
        for _ in range(self.opt_iters):
            optimizer.zero_grad()

            # Returns (B*K)
            costs = self.evaluate(state, actions, task_id)
            returns = -costs
            costs = costs.sum()
            costs.backward()

            # print(actions.grad.size())

            # grad clip
            # Find norm across batch
            if self.grad_clip:
                epsilon = 1e-6
                max_grad_norm = 1.0
                actions_grad_norm = actions.grad.norm(2.0,dim=2,keepdim=True)+epsilon
                # print("before clip", actions.grad.max().cpu().numpy())

                # Normalize by that
                actions.grad.data.div_(actions_grad_norm)
                actions.grad.data.mul_(actions_grad_norm.clamp(min=0, max=max_grad_norm))
                # print("after clip", actions.grad.max().cpu().numpy())

            # print(actions.grad)

            optimizer.step()

            if return_plan_each_iter:
                _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
                best_plan = actions[:, topk[0]].reshape(self.H, self.a_size).detach()
                plan_each_iter.append(best_plan.data.clone())

        actions = actions.detach()
        # Re-fit belief to the K best action sequences
        _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
        best_plan = actions[:, topk[0]].reshape(self.H, self.a_size)

        if return_plan_each_iter:
            return plan_each_iter
        if first_action:
            return best_plan[0]
        else:
            return best_plan