import torch
import numpy as np

from .cem import CEM
from .mppi import MPPI, PDDM
from .lqr import LQR
from .grad import GradPlan
from .manual import Manual
from .reward import GTCost

def quat_mul(q0, q1):
    
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = torch.stack([w, x, y, z], dim=-1)
    q = q / q.norm(2, dim=-1, keepdim=True)
    assert q.shape == q0.shape
    return q


class Agent():
    def __init__(self, hparams):
        self.model_name = hparams.model
        self.env_name = hparams.env
        self.control_dim = hparams.control_dim
        self.state_dim = hparams.state_dim
        self.dnn_out = hparams.dnn_out
        self.reward_discount = hparams.reward_discount
        self._cost = GTCost(self.env_name, self.state_dim, self.control_dim,
            self.reward_discount, hparams.gpuid)

    def act(self, state):
        pass

class RandomAgent(Agent):
    def __init__(self, hparams):
        super(RandomAgent, self).__init__(hparams)
    
    def act(self, state, task_id=None):
        return np.random.randn(self.control_dim, 1)

class MPC(Agent):
    def __init__(self, hparams, model, envs=None, collector=None, likelihood=None, hnet=None):
        super(MPC, self).__init__(hparams)
        self.model = model
        self.hnet = hnet
        self.envs = envs
        self.collector = collector
        self.gpuid = hparams.gpuid
        self.out_var = hparams.out_var
        self.normalize_xu = hparams.normalize_xu if collector is not None else False
        self.gt_dynamic = hparams.gt_dynamic
        self.control_type = hparams.control

        if self.model_name.startswith("hnet") or self.model_name == "chunked_hnet":
            self.reset_hnet()

        if hparams.control == "mpc-cem":
            self.control = CEM(self._dynamics, self._cost, hparams.state_dim, hparams.control_dim, 
                num_samples=hparams.n_sim_particles,
                num_elite=hparams.num_cem_elites,
                num_iterations=hparams.n_sim_steps,
                horizon=hparams.horizon,
                device=hparams.gpuid,
                u_min=None,
                u_max=None,
                choose_best=True,
                init_cov_diag=hparams.mag_noise)
        elif hparams.control == "mpc-mppi":
            noise_sigma = torch.eye(hparams.control_dim, device=hparams.gpuid, dtype=torch.float32) \
                * hparams.mag_noise
            self.control = MPPI(self._dynamics, self._cost, hparams.state_dim, noise_sigma, 
                num_samples=hparams.n_sim_particles,
                num_iter=hparams.n_sim_steps,
                horizon=hparams.horizon,
                lambda_=1/hparams.pddm_kappa,
                device=hparams.gpuid,
                u_min=None,
                u_max=None)
        elif hparams.control == "mpc-pddm":
            self.control = PDDM(self._dynamics, self._cost, hparams.state_dim, hparams.control_dim,
                hparams.horizon, hparams.n_sim_particles, hparams.pddm_beta,
                hparams.pddm_kappa, hparams.mag_noise, hparams.gpuid)
        elif hparams.control == "mpc-grad":
            self.control = GradPlan(self._dynamics, self._cost, hparams.state_dim, hparams.control_dim,
                hparams.n_sim_particles, hparams.n_sim_steps, hparams.horizon, hparams.gpuid)
        elif hparams.control == 'mpc-lqr':
            self.control = LQR(hparams.state_dim, hparams.control_dim, hparams.horizon)
        elif hparams.control == "manual":
            self.control = Manual(hparams.env, hparams.state_dim, hparams.control_dim,
                hparams.horizon, self._dynamics, hparams.gpuid)
    
    def cache_hnet(self, task_id):
        weights = self.hnet(task_id)
        weights = [w.detach() for w in weights]
        self._cached_weights = weights

    def reset_hnet(self):
        self._cached_weights = None

    def cache_state_norm(self, task_id):
        # normalize the state
        if self.normalize_xu:
            x_mu, x_std, a_mu, a_std = self.collector.norm(task_id)
            self.x_mu, self.x_std = x_mu.to(self.gpuid), x_std.to(self.gpuid)
            self.a_mu, self.a_std = a_mu.to(self.gpuid), a_std.to(self.gpuid)

    def _dynamics(self, x, u, task_id):
        x = x.view(-1, self.state_dim)
        u = u.view(-1, self.control_dim)
        xcopy = x.clone()

        # State preprocessing
        if self.env_name.startswith("inverted_pendulum") or self.env_name.startswith('cartpole'):
            x = torch.cat((x[:, 0:1], torch.cos(x[:, 1:2]), torch.sin(x[:, 1:2]), x[:, 2:]), dim=-1)
        elif self.env_name in ["half_cheetah_body", "hopper"]:
            x = torch.cat((x[:, 1:2], torch.cos(x[:, 2:3]), torch.sin(x[:, 2:3]), x[:, 3:]), dim=-1)
        elif self.env_name == "door":
            x = torch.cat((x[:, 0:-1], torch.cos(x[:, -1:]), torch.sin(x[:, -1:])), dim=-1)
        elif self.env_name == "door_pose":
            x = torch.cat((x[:, 0:-2], torch.cos(x[:, -2:-1]), torch.sin(x[:, -2:-1]), 
                    torch.cos(x[:, -1:]), torch.sin(x[:, -1:])), dim=-1)
        # FIXME: REMOVE THIS (now DEBUG ONLY)
        if self.gt_dynamic: 
            if self.env_name == "pendulum":
                th = torch.atan2(x[:, 1], x[:, 0]).view(-1, 1)
                thdot = x[:, 2].view(-1, 1)

                g = 10
                m = 1
                l = 1
                dt = 0.05
                u = torch.clamp(u, -2, 2)

                newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
                newth = th + newthdot * dt
                newthdot = torch.clamp(newthdot, -8, 8)

                cos = torch.cos(newth)
                sin = torch.sin(newth)
                xx_gt = torch.cat((cos, sin, newthdot), dim=1)

        # normalize the state
        if self.normalize_xu:
            x = (x - self.x_mu) / self.x_std
            u = (u - self.a_mu) / self.a_std
         
        if self.model_name in ['single', 'finetune', 'coreset', 'pnn', 'ewc', 'si', 'multitask']:
            xx = self.model(x, u, task_id)

        elif self.model_name.startswith("hnet") or self.model_name == "chunked_hnet":
            weights = self.hnet(task_id) if self._cached_weights is None else self._cached_weights
            xu = torch.cat((x, u), dim=-1)
            xx = self.model.forward(xu, weights)

        # For probablistic output, select the mean
        if self.out_var:
            xx, _ = torch.split(xx, xx.size(-1)//2, dim=-1)

        # output conversion Steps
        # (deprecated) Un-normalize output (typically i'm not normalizing output)
        if self.dnn_out != "diff" and self.normalize_xu:
            xx = xx * self.x_std + self.x_mu

        # Compensate diff
        if self.env_name in ["half_cheetah_body", "hopper"] and self.dnn_out == "diff":
            xx = torch.cat((xx[:, 0:1], xcopy[:, 1:] + xx[:, 1:]), dim=-1)
        elif self.env_name == "door_pose" and self.dnn_out == "diff":
            xx = torch.cat((xcopy[:, 0:3] + xx[:, 0:3], quat_mul(xcopy[:, 3:7], xx[:, 3:7]),
                xcopy[:, 7:] + xx[:, 7:]), dim=-1)
        elif self.dnn_out == "diff":
            xx = xcopy + xx

        if self.gt_dynamic:
            print((xx_gt - xx).mean(dim=0))
            return xx_gt
        return xx
    
    def reset(self):
        self.control.reset()

    def act(self, state, task_id=None, first_action=True):
        self.model.eval()
        if self.control_type != "manual":
            self.cache_state_norm(task_id)

        with torch.no_grad():
            cmd = self.control.command(state, task_id, first_action)
        return cmd

class RollOut():
    def __init__(self, hparams, model, collector):
        self.model = model
        self.collector = collector

        self.n_samples = hparams.n_sim_particles
        self.gpuid = hparams.gpuid

        self.x_dim = hparams.state_dim
        self.a_dim = hparams.control_dim
        self.horizon = hparams.horizon
        self.propagation = hparams.propagation
        self.dnn_out = hparams.dnn_out

    def predict(self, x_t, actions, task_id):
        """
        Multi-step forward, deprecate this
        """
        raise NotImplementedError
    
    def plot_rollout(self, env, x_t, actions, task_id):
        raise NotImplementedError