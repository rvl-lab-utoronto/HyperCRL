import io
import PIL.Image
import torch
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import csv
import pickle
import sys
import recordtype
import json

from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from hypercrl import dataset

from hypercrl import dataset
from hypercrl.envs.cl_env import CLEnvHandler, EnvSpecs

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def np2torch(vec):
    return torch.FloatTensor(vec)

def torch2np(vec):
    return vec.detach().cpu().numpy()

def str_to_act(act_str):
    """Convert the name of an activation function into the actual PyTorch
    activation function.

    Args:
        act_str: Name of activation function (as defined by command-line
            arguments).

    Returns:
        Torch activation function instance or ``None``, if ``linear`` is given.
    """
    if act_str == 'linear':
        act = None
    elif act_str == 'sigmoid':
        act = torch.nn.Sigmoid()
    elif act_str == 'relu':
        act = torch.nn.ReLU()
    elif act_str == 'elu':
        act = torch.nn.ELU()
    else:
        raise Exception('Activation function %s unknown.' % act_str)
    return act

def isfloat(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def read_hparams(folder, file):
    names = []
    values = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['config']
            names.append(name)

            value = row['value']
            if value == "None":
                value = None
            elif value == "True" or value == "TRUE":
                value = True
            elif value == "False" or value == "FALSE":
                value = False
            elif value.isnumeric():
                value = int(value)
            elif isfloat(value):
                value = float(value)
            elif value[0] == '[' and value[-1] == ']':
                value = json.loads(value)

            if name == "save_folder":
                value = folder
            values.append(value)

    # Backward compatability
    if "mlp_var_minmax" not in names:
        names.append("mlp_var_minmax")
        values.append(False)

    Hparams = recordtype('Hparams', names)

    hp = Hparams(*values)
    return hp

class MonitorBase():
    def __init__(self, hparams, model, collector, btest):
        self.eval_every = hparams.eval_every
        self.print_train_every = hparams.print_train_every
        self.log_hist_every = 1000
        self.train_iter = 0
        self.epoch = 0
        self.train_loss = 0
        self.optimizer = None

        self.model = model
        self.model_to_save = {'model': model}
        self.hparams = hparams
        self.collector = collector
        self.btest = btest
        self.tflog_dir = os.path.join(hparams.save_folder, f'TB{hparams.env}_{hparams.model}_{hparams.seed}')
        self.writer = SummaryWriter(log_dir=self.tflog_dir)

        self.val_stats = []

        # For debug model shift
        self.net_param_ckpt = {}
        self.log_hparams()

    def log_hparams(self):
        hp_dict = self.hparams.__dict__
        with open(f'{self.tflog_dir}/hparams.csv', 'w') as f:
            fieldnames = ['config', 'value']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for key, val in hp_dict.items():
                if val is None:
                    val = "None"
                if isinstance(val, list):
                    val = str(val)
                writer.writerow({'config': key, 'value': val})

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_step(self, loss):
        self.train_loss += loss.data
        if (self.train_iter % self.print_train_every == 0):
            self.train_loss /= self.print_train_every
            self.writer.add_scalar('train/mse_loss', self.train_loss, self.train_iter)

            print(f"Batch: {self.train_iter}, Loss: {self.train_loss.item():.5f}")
            self.train_loss = 0

        if (self.train_iter % self.log_hist_every == 0):
            diff = 0
            for name, param in self.model.named_parameters():
                if name in self.net_param_ckpt:
                    param = param.detach().cpu()
                    diff += torch.norm(param - self.net_param_ckpt[name], p = 2)

                self.writer.add_histogram(f'train/weight/{name}', param.flatten(), self.train_iter)

            self.writer.add_scalar('train/regularizer', diff, self.train_iter)
        
        self.train_iter += 1

    def log_weight(self):
        for name, param in self.model.named_parameters():
            self.net_param_ckpt[name] = param.detach().cpu()

    def validate(self, mll):
        if (self.train_iter % self.eval_every) == 0:
            self.model.eval()

            bs = self.hparams.bs
            num_tasks = self.collector.num_tasks()

            for i in range(num_tasks):
                is_training = (i == (num_tasks - 1))
                # Only evaluate current task in single task model
                if self.hparams.model == "single" and (not is_training):
                    continue
                if len(self.val_stats) <= i:
                    self.val_stats.append({"time": [],
                        "nll": [], "diff": []})
                _, val_sets = self.collector.get_dataset(i)
                loader = DataLoader(val_sets, batch_size = bs,
                    num_workers=self.hparams.num_ds_worker)

                # Determine if we are validating the currently training task
                val_nll, val_diff = self.validate_task(i, loader, mll, is_training)
                
                self.val_stats[i]['time'].append(self.train_iter)
                self.val_stats[i]['nll'].append(val_nll.item())
                self.val_stats[i]['diff'].append(val_diff.mean().item())

            self.model.train()
            # Other Sfuff
            # self.btest.plot()

    def validate_task(self, task_id, loader, mll, is_training=False):
        gpuid = self.hparams.gpuid

        # Initialize Stats
        val_loss = 0
        val_diff = 0
        N = len(loader)

        with torch.no_grad():
            for _, data in enumerate(loader):
                x_t, a_t, x_tt = data
                x_t, a_t, x_tt = x_t.to(gpuid), a_t.to(gpuid), x_tt.to(gpuid)

                if is_training:
                    # Inference in weight space
                    output = self.model(x_t, a_t)
                else:
                    # Inference in function space
                    output = self.model(x_t, a_t, task_id=task_id)

                loss = -mll(output, x_tt, task_id=task_id)
                if self.hparams.out_var:
                    output, _ = torch.split(output, output.size(-1)//2, dim=-1)
                diff = torch.abs(output - x_tt).mean(dim=0)

                val_loss += loss
                val_diff += diff

            val_loss = val_loss / N
            val_diff = val_diff / N

        print(f"Iter {self.train_iter}, Task: {task_id}, " + \
              f"Val Loss: {val_loss.item():.5f}, Val Diff: {val_diff.mean().item()}")

        return val_loss, val_diff

    def plot(self):
        plt.ioff()
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(1, 2)

        plt.subplot(gs[0, 0])
        for i, stats in enumerate(self.val_stats):
            diff = stats['diff']
            time = stats['time']
            plt.plot(time, diff, label=f'task_{i+1}')
        plt.xlabel("Step")
        plt.ylabel("L1 Diff")
        plt.ylim(0, 0.2)
        plt.title(self.hparams.model)
        plt.legend()

        plt.subplot(gs[0, 1])
        for i, stats in enumerate(self.val_stats):
            diff = stats['nll']
            time = stats['time']
            plt.plot(time, diff, label=f'task_{i+1}')
        plt.xlabel("Step")
        plt.ylabel("NLL")
        plt.ylim(-13, 5)
        plt.title(self.hparams.model)
        plt.legend()
        plt.show()

    def save(self):
        hp = self.hparams
        # Save training csv
        with open(f'{hp.save_folder}/{hp.env}_{hp.model}_{hp.seed}.csv', 'w') as f:
            fieldnames = ['task', 'time', 'diff']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            
            for task_id, stats in enumerate(self.val_stats):
                diff = stats['diff']
                time = stats['time']
                rows = [{'task': task_id, 'time': t, 'diff': d} for (t, d) in zip(time, diff)]
                
                writer.writerows(rows)


class MonitorRL(MonitorBase):
    def __init__(self, hparams, agent, model, collector, btest):
        super(MonitorRL, self).__init__(hparams, model, collector, btest)
        self.num_envs = hparams.num_tasks
        self.agent = agent

        self.env_iter = 0
        self.eval_env_run_every = hparams.eval_env_run_every
        self.run_eval_env_eps = hparams.run_eval_env_eps
        self.rl_stats = [{"step": [], "reward": [], "time": [], "diff": []} for _ in range(self.num_envs)]
        self.rewards = []

        self.eval_envs = CLEnvHandler(hparams.env, hparams.seed)
        for task_id in range(hparams.num_tasks):
            self.eval_envs.add_task(task_id, render=False)

    def env_step(self, state, reward, done, info, task_id):
        self.env_iter += 1
        self.rewards.append(reward)
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            self.writer.add_scalar(f'rl/task_{task_id}/reward', eprew, self.env_iter)
            self.writer.add_scalar(f'rl/task_{task_id}/ep_len', eplen, self.env_iter)
            print(f"Step: {self.env_iter}, Reward: {eprew:.3f}, Episode Length {eplen}")
            self.rewards = []

        if self.env_iter % self.eval_env_run_every == 0:
            self.run_eval_env(task_id)
        
        # Log dataset norm statistic
        if self.env_iter % self.eval_env_run_every == 0:
            self.log_xu_norm(task_id)

    def log_xu_norm(self, task_id):
        if self.hparams.normalize_xu == False:
            return
        
        x_mu, x_std, a_mu, a_std = self.collector.norm(task_id)
        self.writer.add_histogram(f'rl/task_{task_id}/x_mu', x_mu, self.env_iter)
        self.writer.add_histogram(f'rl/task_{task_id}/x_std', x_std, self.env_iter)
        self.writer.add_histogram(f'rl/task_{task_id}/a_mu', a_mu, self.env_iter)
        self.writer.add_histogram(f'rl/task_{task_id}/a_std', a_std, self.env_iter)

    def run_eval_env(self, task_id):

        def model_rollout(env, x_t, agent, tid, plot=False):
            gpuid = self.hparams.gpuid
            x_dim = self.hparams.state_dim
            horizon = self.hparams.horizon

            x_model = torch.tensor(x_t, dtype=torch.float32, device=gpuid).view(-1, x_dim)

            gt_states = [x_t]
            model_states = [x_model]

            # Agent run MPC on this initial state
            agent.reset()
            actions = agent.act(x_model, task_id=tid, first_action=False)
            # Agent evaluate dynamics on this set up mpc action sequences
            for t in range(horizon):
                u = actions[t]
                x_t, _, _, _ = env.step(u.cpu().numpy().reshape(env.action_space.shape))
                gt_states.append(x_t)

                with torch.no_grad():
                    x_model = agent._dynamics(x_model, u, tid)
                model_states.append(x_model)
            
            gt_states = np.stack(gt_states)
            model_states = torch.cat(model_states, dim=0).cpu().numpy()

            if plot:
                times = np.arange(horizon + 1)
                nrows = (x_dim + 1) // 2
                fig = plt.figure(figsize=(10, 6))
                names = EnvSpecs.get_dim_name(self.hparams.env)
                units = EnvSpecs.get_dim_unit(self.hparams.env)
                for d in range(x_dim):
                    fig.add_subplot(nrows, 2, d + 1)
                    plt.plot(times, gt_states[:, d], label='gt')
                    plt.plot(times, model_states[:, d], label='model')
                    plt.title(names[d])
                    plt.ylabel(units[d])
                    plt.xlabel("Planning Horizon")

                fig.suptitle('Open Loop Dynamics')
                plt.legend()
                plt.tight_layout()
                self.writer.add_figure(f'task_{tid}/rollout', fig, self.env_iter)
                plt.close(fig)
            return gt_states, model_states

        def plot_diff_hist(traj_diff):
            horizon, x_dim = traj_diff.shape
            nrows = (x_dim + 1) // 2
            if x_dim > 10:
                fig = plt.figure(figsize=(10, 12))
            elif x_dim >= 20:
                fig = plt.figure(figsize=(10, 18))
            else:
                fig = plt.figure(figsize=(10, 6))
            names = EnvSpecs.get_dim_name(self.hparams.env)
            units = EnvSpecs.get_dim_unit(self.hparams.env)
            for d in range(x_dim):
                times = np.arange(horizon)
                fig.add_subplot(nrows, 2, d + 1)
                plt.bar(times, traj_diff[:, d])
                plt.title(names[d])
                plt.ylabel(units[d])
                plt.xlabel("Planning Horizon")

            fig.suptitle('Open Loop Dynamics (averged diff over 10 init states)')
            plt.tight_layout()
            self.writer.add_figure(f'task_{tid}/rollout_avg_diff', fig, self.env_iter)
            plt.close(fig)

        def run_one_eps(env, agent, tid):
            ts = time.time()
            done = False
            x_t = env.reset()
            self.agent.reset()

            rewards = []
            xs, us = [], []
            while (not done):
                #env.render()
                u_t = agent.act(x_t, task_id=tid).cpu().numpy()
                x_tt, reward, done, info = env.step(u_t.reshape(env.action_space.shape))
                xs.append(x_t)
                us.append(u_t)
                x_t = x_tt
                rewards.append(reward)
            xs.append(x_tt)
            eprew = np.sum(rewards)
            if self.hparams.env == "metaworld10":
                print(f"Task {env.active_task}, Success {info['success']}")
            tdone = time.time() - ts
            return eprew, xs, us, tdone

        for tid in range(1 + task_id):
            env = self.eval_envs.get_env(tid)
            if self.hparams.model == "single" and tid != task_id:
                continue
            if self.hparams.model.startswith("hnet") or self.hparams.model == "chunked_hnet":
                self.agent.cache_hnet(tid)
            self.agent.cache_state_norm(tid)
            
            # # Evaluate prediction dyanmic differences
            # if self.hparams.model != "gt":
            #     x_t = env.reset()
            #     gt_states, model_states = model_rollout(env, x_t, self.agent, tid, plot=True)

            # Run one evaluation episode using learned policy
            stats = [run_one_eps(env, self.agent, tid) for _ in range(self.run_eval_env_eps)]
            eprews, xs, us, times = zip(*stats)
            eprew = np.mean(eprews)
            mean_time = np.mean(times)

            # Collect Reward
            self.rl_stats[tid]['step'].append(self.env_iter)
            self.rl_stats[tid]['reward'].append(eprew)
            self.rl_stats[tid]['time'].append(mean_time)

            # Evaluate multi-step prediction difference on state/act distribution based on current MPC
            traj_diff = []
            with torch.no_grad():
                xs = torch.tensor(xs, dtype=torch.float32, device=self.hparams.gpuid)
                us = torch.tensor(us, dtype=torch.float32, device=self.hparams.gpuid)
                num = xs.size(1) - self.hparams.horizon # number of traj in an episode

                # Select all possible initial states
                states = xs[:, 0:num, :].reshape(-1, self.hparams.state_dim)
                for t in range(self.hparams.horizon):
                    # select the current action
                    actions = us[:, t:t+num, :].reshape(-1, self.hparams.control_dim)
                    preds = self.agent._dynamics(states, actions, tid)
                    
                    # compare to the ground truth
                    gt = xs[:, t+1:t+1+num, :].reshape(-1, self.hparams.state_dim)
                    traj_diff.append(torch.abs(gt-preds).mean(dim=0))
                    
                    # Keep making predictions
                    states = preds

            traj_diff = torch.stack(traj_diff, dim=0).detach().cpu().numpy()
            # Plot a histogram
            plot_diff_hist(traj_diff)

            # Save for later
            l1_pred_diff = traj_diff[0].mean()
            self.rl_stats[tid]['diff'].append(l1_pred_diff)

            print(f"Task: {tid}, Step: {self.env_iter} Eval reward: {eprew:.3f}, " +
                 f"On-policy Diff: {l1_pred_diff:.5f}, Time Taken {mean_time:.1f}s")

    def save(self, task_id):
        super(MonitorRL, self).save()

        hp = self.hparams
        with open(f'{hp.save_folder}/RL{hp.env}_{hp.model}_{hp.seed}.csv', 'w') as f:
            fieldnames = ['task', 'envstep', 'reward', 'runtime', 'on_policy_diff']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            for tid, stats in enumerate(self.rl_stats):
                time = stats['time']
                reward = stats['reward']
                step = stats['step']
                diff = stats['diff']
                rows = [{'task': tid,
                         'runtime': t,
                         'reward': r,
                         'envstep': s,
                         'on_policy_diff': d}
                        for (t, r, s, d) in zip(time, reward, step, diff)]

                writer.writerows(rows)

        # Save Model
        save_dict = {'train_iter': self.train_iter,
            'env_step': self.env_iter,
            'num_tasks_seen': self.collector.num_tasks(),
        }
        for name, model in self.model_to_save.items():
            save_dict[f'{name}_state_dict'] = model.state_dict()

        if self.optimizer is not None:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        # Save the model twice. One permenant, one cached
        torch.save(save_dict, f'{self.tflog_dir}/model.pt')
        torch.save(save_dict, f'{self.tflog_dir}/model_{task_id}.pt')

        # Save Data Collector
        with open(f'{self.tflog_dir}/data.pkl', 'wb') as f:
            pickle.dump(self.collector, f, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def resume_from_disk(hparams):
        tflog_dir = os.path.join(hparams.save_folder, f'TB{hparams.env}_{hparams.model}_{hparams.seed}')
        with open(f'{tflog_dir}/data.pkl', 'rb') as f:
            # For backward compatability
            sys.modules['dataset'] = dataset
            collector = pickle.load(f)

        return collector

    def load_stats(self, checkpoint):
        self.env_iter = checkpoint['env_step']
        self.train_iter = checkpoint['train_iter']
        num_tasks_seen = checkpoint['num_tasks_seen']

        # Restore rl stats object
        hp = self.hparams
        with open(f'{hp.save_folder}/RL{hp.env}_{hp.model}_{hp.seed}.csv', 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                task = int(row['task'])
                reward = float(row['reward'])
                envstep = int(row['envstep'])
                runtime = float(row['runtime'])
                diff = float(row['on_policy_diff'])
                self.rl_stats[task]['step'].append(envstep)
                self.rl_stats[task]['reward'].append(reward)
                self.rl_stats[task]['time'].append(runtime)
                self.rl_stats[task]['diff'].append(diff)

        # Restore val stats object
        for _ in range(num_tasks_seen):
            self.val_stats.append({"time": [], "nll": [], "diff": []})

        with open(f'{hp.save_folder}/{hp.env}_{hp.model}_{hp.seed}.csv', 'r') as f:
            reader = csv.DictReader(f)
        
            self.val_stats.append({"time": [],
                        "nll": [], "diff": []})
            for row in reader:
                task = int(row['task'])
                diff = float(row['diff'])
                time = float(row['time'])
                self.val_stats[task]['diff'].append(diff)
                self.val_stats[task]['time'].append(time)