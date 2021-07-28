import numpy as np
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import torch
import os

from torch.utils.data import DataLoader

from hypercrl.tools import reset_seed, str_to_act
from hypercrl.tools import MonitorHnet, HP, Hparams
from hypercrl.control import RandomAgent, MPC
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.dataset.datautil import DataCollector

from hypercrl.model import build_model_hnet as build_model
from hypercrl.model import reload_model_hnet as reload_model
from hypercrl.hypercl.utils import hnet_regularizer as hreg
from hypercrl.hypercl.utils import ewc_regularizer as ewc
from hypercrl.hypercl.utils import si_regularizer as si
from hypercrl.hypercl.utils import optim_step as opstep


class TaskLoss(torch.nn.Module):
    def __init__(self, hparams, mnet):
        super(TaskLoss, self).__init__()
        self.out_var = hparams.out_var
        self.y_dim = hparams.out_dim
        self.mlp_var_minmax = hparams.mlp_var_minmax

        self.reg_norm = 1
        self.reg_lambda = hparams.reg_lambda
        self.mnet = mnet

    def regularize(self, weights):
        if self.reg_lambda == 0:
            return 0

        loss = 0
        for weight in weights:
            loss = weight.norm(self.reg_norm)

        return loss

    def reg_logvar(self, weights):
        if self.mlp_var_minmax:
            max_logvar = self.mnet.mlp_max_logvar
            min_logvar = self.mnet.mlp_min_logvar
        else:
            max_logvar = weights[0]
            min_logvar = weights[1]

        # Regularize max/min var
        loss = 0.01 * (max_logvar.sum() - min_logvar.sum())
        return loss

    def forward(self, pred, gt, weights, add_reg_logvar=True):
        if self.out_var:
            mu, logvar = torch.split(pred, self.y_dim, dim=-1)

            # Compute loss of a task (i.e during evaluation)
            inv_var = torch.exp(-logvar)
            loss = ((mu - gt) ** 2) * inv_var + logvar
            loss = loss.sum() / self.y_dim

            if add_reg_logvar:
                loss += self.reg_logvar(weights)

        else:
            loss = torch.nn.functional.mse_loss(pred, gt, reduction='sum')
            loss = loss / self.y_dim

        loss += self.regularize(weights)

        return loss


class TaskLossMT(TaskLoss):
    def __init__(self, hparams, mnet, hnet, collector, task_id):
        super().__init__(hparams, mnet)
        self.hnet = hnet
        self.task_id = task_id
        self.gpuid = hparams.gpuid

        self.add_trainset(collector, task_id, hparams)

    def add_trainset(self, collector, task_id, hparams):
        old_data, old_data_iter = [], []
        for tid in range(0, task_id):
            train_set, _ = collector.get_dataset(tid)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.bs,
                                                       shuffle=True, drop_last=True)
            old_data.append(train_loader)
            old_data_iter.append(iter(train_loader))

        self.old_data = old_data
        self.old_data_iter = old_data_iter

    def replay(self, tid):
        loader_it = self.old_data_iter[tid]
        try:
            data = next(loader_it)
        except StopIteration:
            # Reset the dataloader iterable
            loader_it = iter(self.old_data[tid])
            self.old_data_iter[tid] = loader_it
            data = next(loader_it)

        x_t, a_t, x_tt = data
        x_t, a_t, x_tt = x_t.to(self.gpuid), a_t.to(self.gpuid), x_tt.to(self.gpuid)

        # Forward Pass
        X = torch.cat((x_t, a_t), dim=-1)
        weights = self.hnet.forward(tid)
        Y = self.mnet.forward(X, weights)

        # Task-specific loss.
        loss_task = super().forward(Y, x_tt, weights, add_reg_logvar=False)
        return loss_task

    def forward(self, pred, gt, weights):
        loss = super().forward(pred, gt, weights)
        for tid in range(0, self.task_id):
            loss += self.replay(tid)
        return loss


class TaskLossReplay(TaskLossMT):
    def __init__(self, hparams, mnet, hnet, collector, task_id):
        super().__init__(hparams, mnet, hnet, collector, task_id)

    def add_trainset(self, collector, task_id, hparams):
        old_data, old_data_iter = [], []
        M = hparams.bs // task_id if task_id > 0 else 0
        for tid in range(0, task_id):
            train_set, _ = collector.get_dataset(tid)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=M,
                                                       shuffle=True, drop_last=True)
            old_data.append(train_loader)
            old_data_iter.append(iter(train_loader))

        self.old_data = old_data
        self.old_data_iter = old_data_iter


def augment_model(task_id, mnet, hnet, collector, hparams):
    # Regularizer targets.
    targets = hreg.get_current_targets(task_id, hnet)

    # Add new hypernet embeddings and Loss Function
    hnet.add_task(task_id, hparams.std_normal_temb)

    if hparams.model == "hnet_mt":
        # Loss Function
        mll = TaskLossMT(hparams, mnet, hnet, collector, task_id)
    elif hparams.model == "hnet_replay":
        mll = TaskLossReplay(hparams, mnet, hnet, collector, task_id)
    else:
        mll = TaskLoss(hparams, mnet)

    # (Re)Put model to GPU
    gpuid = hparams.gpuid
    mnet.to(gpuid)
    hnet.to(gpuid)

    # Optimize over the GP model params and likelihood param

    mnet.train()
    hnet.train()

    # Collect Fisher estimates for the reg computation.
    fisher_ests = None
    if hparams.ewc_weight_importance and task_id > 0:
        fisher_ests = []
        n_W = len(hnet.target_shapes)
        for t in range(task_id):
            ff = []
            for i in range(n_W):
                _, buff_f_name = ewc._ewc_buffer_names(t, i, False)
                ff.append(getattr(mnet, buff_f_name))
            fisher_ests.append(ff)

    # Register SI buffers for new task
    si_omega = None
    if hparams.model == "hnet_si":
        si.si_register_buffer(mnet, hnet, task_id)
        if task_id > 0:
            si_omega = si.get_si_omega(mnet, task_id)

    regularized_params = list(hnet.theta)
    if task_id > 0 and hparams.plastic_prev_tembs:
        for i in range(task_id):  # for all previous task embeddings
            regularized_params.append(hnet.get_task_emb(i))
    theta_optimizer = torch.optim.Adam(regularized_params, lr=hparams.lr_hyper)
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    emb_optimizer = torch.optim.Adam([hnet.get_task_emb(task_id)],
                                     lr=hparams.lr_hyper)

    trainer_misc = (targets, mll, theta_optimizer, emb_optimizer, regularized_params,
                    fisher_ests, si_omega)

    return trainer_misc


def augment_model_after(task_id, mnet, hnet, hparams, collector):
    if hparams.model == "hnet_si":
        si.update_omega(mnet, hnet, hparams.si_eps, task_id)

    if hparams.ewc_weight_importance:
        ## Estimate Fisher for outputs of the hypernetwork.
        weights = hnet.forward(task_id)

        # Note, there are actually no parameters in the main network.
        fake_main_params = torch.nn.ParameterList()
        for i, W in enumerate(weights):
            fake_main_params.append(torch.nn.Parameter(torch.Tensor(*W.shape),
                                                       requires_grad=True))
            fake_main_params[i].data = weights[i]

        ewc.compute_fisher(task_id, collector, fake_main_params, hparams.gpuid, mnet,
                           empirical_fisher=True, online=False, n_max=hparams.n_fisher,
                           regression=True, allowed_outputs=None, out_var=hparams.out_var)


def train(task_id, mnet, hnet, trainer_misc, logger, train_set, hparams):
    # Data Loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.bs, shuffle=True,
                                               drop_last=True, num_workers=hparams.num_ds_worker)

    # GPUID
    gpuid = hparams.gpuid

    regged_outputs = None

    targets, mll, theta_optimizer, emb_optimizer, regularized_params, \
    fisher_ests, si_omega = trainer_misc

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and hparams.beta > 0

    it = 0
    while it < hparams.train_dynamic_iters:
        mnet.train()
        hnet.train()
        for i, data in enumerate(train_loader):
            if len(data) == 3:
                x_t, a_t, x_tt = data
                x_t, a_t, x_tt = x_t.to(gpuid), a_t.to(gpuid), x_tt.to(gpuid)
                X = torch.cat((x_t, a_t), dim=-1)
            else:
                X, x_tt = data
                X, x_tt = X.to(gpuid), x_tt.to(gpuid)

            ### Train theta and task embedding.
            theta_optimizer.zero_grad()
            emb_optimizer.zero_grad()

            weights = hnet.forward(task_id)
            if hparams.model == "hnet_si":
                si.si_update_optim_step(mnet, weights, task_id)
                for weight in weights:
                    weight.retain_grad()  # save grad for calculate si path integral

            Y = mnet.forward(X, weights)
            # Task-specific loss.
            loss_task = mll(Y, x_tt, weights)
            # We already compute the gradients, to then be able to compute delta
            # theta.
            loss_task.backward(retain_graph=calc_reg,
                               create_graph=hparams.backprop_dt and calc_reg)
            torch.nn.utils.clip_grad_norm_(hnet.get_task_emb(task_id), hparams.grad_max_norm)

            # The task embedding is only trained on the task-specific loss.
            # Note, the gradients accumulated so far are from "loss_task".
            emb_optimizer.step()

            # SI
            if hparams.model == "hnet_si":
                torch.nn.utils.clip_grad_norm_(weights, hparams.grad_max_norm)
                si.si_update_grad(mnet, weights, task_id)

            # Update Regularization
            loss_reg = torch.tensor(0., requires_grad=False)
            dTheta = None
            grad_tloss = None
            if calc_reg:
                if i % 1000 == 0:  # Just for debugging: displaying grad magnitude.
                    grad_tloss = torch.cat([d.grad.clone().view(-1) for d in
                                            hnet.theta])
                if hparams.no_look_ahead:
                    dTheta = None
                else:
                    dTheta = opstep.calc_delta_theta(theta_optimizer,
                                                     hparams.use_sgd_change, lr=hparams.lr_hyper,
                                                     detach_dt=not hparams.backprop_dt)

                if hparams.plastic_prev_tembs:
                    dTembs = dTheta[-task_id:]
                    dTheta = dTheta[:-task_id] if dTheta is not None else None
                else:
                    dTembs = None

                loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                                                    targets=targets, dTheta=dTheta, dTembs=dTembs, mnet=mnet,
                                                    inds_of_out_heads=regged_outputs,
                                                    fisher_estimates=fisher_ests,
                                                    si_omega=si_omega)

                loss_reg = loss_reg * hparams.beta * Y.size(0)

                loss_reg.backward()

                if grad_tloss is not None:  # Debug
                    grad_full = torch.cat([d.grad.view(-1) for d in hnet.theta])
                    # Grad of regularizer.
                    grad_diff = grad_full - grad_tloss
                    grad_diff_norm = torch.norm(grad_diff, 2)

                    # Cosine between regularizer gradient and task-specific
                    # gradient.
                    if dTheta is None:
                        dTheta = opstep.calc_delta_theta(theta_optimizer,
                                                         hparams.use_sgd_change, lr=hparams.lr_hyper,
                                                         detach_dt=not hparams.backprop_dt)
                    dT_vec = torch.cat([d.view(-1).clone() for d in dTheta])
                    grad_cos = torch.nn.functional.cosine_similarity(grad_diff.view(1, -1),
                                                                     dT_vec.view(1, -1))

                    grad_tloss = (grad_tloss, grad_full, grad_diff_norm, grad_cos)

            torch.nn.utils.clip_grad_norm_(regularized_params, hparams.grad_max_norm)
            theta_optimizer.step()

            logger.train_step(loss_task, loss_reg, dTheta, grad_tloss, weights)
            # Validate
            logger.validate(mll)

            it += 1
            if it >= hparams.train_dynamic_iters:
                break


def plot_embs(hparams, embs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for emb in embs:
        emb = emb.detach().cpu().numpy()
        ax.plot(emb[0], emb[1], 'kx')
    fig.savefig(f'{hparams.save_folder}/embedding_{hparams.seed}.png')


def play_model(hparams, runs=10):
    _, hnet, agent, checkpoint, _ = reload_model(hparams)

    # Reset seed
    reset_seed(hparams.seed)

    # Task Embedding
    embs = hnet.get_task_embs()
    # plot_embs(hparams, embs)

    envs = CLEnvHandler(hparams.env, hparams.robot, hparams.seed)
    for task_id in range(checkpoint['num_tasks_seen']):
        # Cache the mainnet weight
        agent.cache_hnet(task_id)
        env = envs.add_task(task_id, render=True)

        avg_rewards = []
        for _ in range(runs):
            rewards = []
            x_t = env.reset()
            agent.reset()
            done = False
            while (not done):
                env.render()
                u_t = agent.act(x_t, task_id=task_id).cpu().numpy()
                x_tt, reward, done, _ = env.step(u_t.reshape(env.action_space.shape))
                x_t = x_tt
                rewards.append(reward)
            eprew = np.sum(rewards)
            avg_rewards.append(eprew)
            print(f"Task {task_id + 1}, episode reward {eprew}, ep length {len(rewards)}")

        avg_reward = np.mean(avg_rewards)
        print(f"Average reward for task {task_id + 1} is {avg_reward}")
        env.close()


def run(hparams, render=False):
    print(f'Running')

    # Reset seed
    reset_seed(hparams.seed)

    # Fix data/env seed for lqr10
    if hparams.env == "lqr10" and hparams.rand_aggregate_seed is not None:
        np.random.seed(hparams.rand_aggregate_seed)
        random.seed(hparams.rand_aggregate_seed)

    if hparams.resume:
        # Restore model and agent
        mnet, hnet, agent, checkpoint, collector = reload_model(hparams)

        # Restore Logger
        logger = MonitorHnet(hparams, agent, mnet, hnet, collector)
        logger.load_stats(checkpoint)

        # Get num tasks we previously trained
        num_tasks_seen = checkpoint['num_tasks_seen']

    else:
        # Collect some random data
        collector = DataCollector(hparams)

        # Build model
        mnet, hnet = build_model(hparams)

        # RL Agent
        agent = MPC(hparams, mnet, collector=collector, hnet=hnet)

        # Monitor
        logger = MonitorHnet(hparams, agent, mnet, hnet, collector)

        # Start from scratch
        num_tasks_seen = 0

    # Convert to cuda
    mnet.to(hparams.gpuid)
    hnet.to(hparams.gpuid)

    # Random Policy
    rand_pi = RandomAgent(hparams)

    # Start learning in environment
    envs = CLEnvHandler(hparams.env, hparams.robot, hparams.seed)
    if hparams.resume:
        for tid in range(num_tasks_seen):
            envs.add_task(tid)

    for task_id in range(num_tasks_seen, hparams.num_tasks):
        # New Task with different friction
        env = envs.add_task(task_id, render=render)

        print(f"Collecting some random data first for task {task_id}")
        x_t = env.reset()
        for it in range(hparams.init_rand_steps):
            u = rand_pi.act(x_t)
            x_tt, _, done, _ = env.step(u.reshape(env.action_space.shape))
            collector.add(x_t, u, x_tt, task_id)
            x_t = x_tt
            logger.data_aggregate_step(x_tt, task_id, it)
            if done:
                x_t = env.reset()

        # Augment Model, instantiate optimizers/regularizer targets
        trainer_misc = augment_model(task_id, mnet, hnet, collector, hparams)

        # Interact with the environment
        x_t = env.reset()
        agent.reset()
        for it in range(hparams.max_iteration):
            if it % hparams.dynamics_update_every == 0:
                # Train Dynamics Model
                ts = time.time()
                train_set, _ = collector.get_dataset(task_id)
                train(task_id, mnet, hnet, trainer_misc, logger, train_set, hparams)
                print(f"Training time", time.time() - ts)

            if render:
                env.render()
            # Cache the mainnet weight
            agent.cache_hnet(task_id)
            # Run MPC
            u_t = agent.act(x_t, task_id=task_id).detach().cpu().numpy()
            x_tt, reward, done, info = env.step(u_t.reshape(env.action_space.shape))

            # Update the dataset of the env in which we're training 
            collector.add(x_t, u_t, x_tt, task_id)
            x_t = x_tt

            if done:
                x_t = env.reset()
                agent.reset()

            logger.env_step(x_tt, reward, done, info, task_id)

        augment_model_after(task_id, mnet, hnet, hparams, collector)

        # Save Model
        logger.save(task_id)

    envs.close()
    logger.writer.close()


def chunked_hnet(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "chunked_hnet"

    hparams = Hparams.add_chunked_hnet_hparams(hparams)

    if play:
        play_model(hparams)
    else:
        run(hparams)


def hnet(env, robot="Panda", seed=None, savepath=None, resume=False, render=False, play=False, runs=10):
    # Hyperparameters
    hparams = HP(env=env, robot=robot, seed=seed, save_folder=savepath, resume=resume)
    hparams.model = "hnet"

    hparams = Hparams.add_hnet_hparams(hparams)

    if play:
        play_model(hparams, runs)
    else:
        run(hparams, render)


def hnet_si(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "hnet_si"

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0.05
    hparams.grad_max_norm = 5

    if play:
        play_model(hparams)
    else:
        run(hparams)


def hnet_ewc(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "hnet_ewc"

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0.05
    hparams.ewc_weight_importance = True
    hparams.n_fisher = -1

    if play:
        play_model(hparams)
    else:
        run(hparams)


def hnet_mt(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "hnet_mt"

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0
    hparams.plastic_prev_tembs = True

    if play:
        play_model(hparams)
    else:
        run(hparams)


def hnet_replay(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "hnet_replay"

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0.05
    hparams.grad_max_norm = 5
    hparams.plastic_prev_tembs = True

    if play:
        play_model(hparams)
    else:
        run(hparams)


if __name__ == "__main__":
    import fire

    fire.Fire({
        'hnet': hnet,
        'chunked_hnet': chunked_hnet,
        'hnet_si': hnet_si,
        'hnet_ewc': hnet_ewc,
    })
