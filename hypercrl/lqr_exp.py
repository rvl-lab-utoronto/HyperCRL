import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import torch
import os

from torch.utils.data import TensorDataset, DataLoader

from hypercrl.model.mbrl import (MSELoss, LogProbLoss,
                        IPSelector, BoundaryTest)
from hypercrl.model.regularizer import EWCLoss, SILoss
from hypercrl.model import build_model, reload_model
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.tools import reset_seed, np2torch, torch2np
from hypercrl.tools import MonitorRL, HP
from hypercrl.control import RandomAgent, MPC, RollOut
from hypercrl.dataset.datautil import DataCollector, train_val_split


def augment_model_before(task_id, model, logger, collector, hparams):
    # Add Weight and define loss
    if hparams.model == "pnn":
        loss_fn = LogProbLoss if hparams.out_var else MSELoss
        mll = loss_fn(model, task_id=task_id, reg_lambda = hparams.reg_lambda)
        model.add_weights(task_id)
    elif hparams.model == "coreset":
        model.add_weights(task_id)
        loss_fn = LogProbLoss if hparams.out_var else MSELoss
        mll = loss_fn(model, task_id=task_id, reg_lambda = hparams.reg_lambda, M=hparams.M)
    elif hparams.model == "multitask":
        if hparams.mt_reinit and task_id > 0:
            model.reinit()
        model.add_weights(task_id)
        loss_fn = LogProbLoss if hparams.out_var else MSELoss
        mll = loss_fn(model, task_id=task_id, reg_lambda = hparams.reg_lambda, M=hparams.bs)
    elif hparams.model == "finetune":
        model.add_weights(task_id)
        loss_fn = LogProbLoss if hparams.out_var else MSELoss
        mll = loss_fn(model, task_id=-1, reg_lambda=hparams.reg_lambda)
    elif hparams.model == "single":
        if task_id > 0:
            model.reinit()
        model.add_weights(task_id)
        loss_fn = LogProbLoss if hparams.out_var else MSELoss
        mll = loss_fn(model, task_id=-1, reg_lambda=hparams.reg_lambda)        
    elif hparams.model == "ewc":
        model.add_weights(task_id)
        mll = EWCLoss(model, task_id=task_id, reg_lambda = hparams.reg_lambda,
                ewc_beta=hparams.ewc_beta, out_var=hparams.out_var)
    elif hparams.model == "si":
        model.add_weights(task_id)
        # Zero si weight importance
        model.to(hparams.gpuid)
        model.si_zero_stats() 
        mll = SILoss(model, task_id=task_id, reg_lambda = hparams.reg_lambda,
                si_c=hparams.si_c, out_var=hparams.out_var)

    # Put model to GPU
    gpuid = hparams.gpuid
    model.to(gpuid)

    # Optimize over the GP model params
    optimizer = torch.optim.Adam(list(model.parameters()),
                                lr=hparams.lr)
    # LR schedular
    if hparams.lr_steps is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, hparams.lr_steps, 0.1)
    else:
        scheduler = None

    logger.set_optimizer(optimizer)
    trainer = optimizer, scheduler, mll

    return trainer

def augment_model_after(task_id, model, logger, collector, hparams):
    if hparams.model == "coreset":
        # Inducing Point Selection
        train_set, _ = collector.get_dataset(task_id)
        ip_selector = IPSelector(train_set, hparams)
        inducing_points = ip_selector.inducing_points()

        model.add_inducing_points(inducing_points, task_id)
        model.freeze(task_id)
        # Log the weight of the network after a task is done
        logger.log_weight()
    elif hparams.model == "pnn":
        # weight freezing
        model.freeze(task_id)

        model.add_inducing_points(inducing_points, q_u, task_id)
    elif hparams.model == "ewc":
        train_set, _ = collector.get_dataset(task_id)
        model.estimate_fisher(train_set, allowed_classes=None, collate_fn=None)
    elif hparams.model == "si":
        model.update_omega(hparams.si_epsilon)
    elif hparams.model == "multitask":
        model.add_trainset(collector, task_id)
    return model

def train(task_id, model, trainer, logger, collector, btest, hparams):

    # Data Loader
    train_set, val_set = collector.get_dataset(task_id)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.bs, shuffle=True,
                    drop_last=True, num_workers=hparams.num_ds_worker)

    # loss, optimizer, scheduler
    optimizer, scheduler, mll = trainer

    gpuid = hparams.gpuid

    it = 0
    while it < hparams.train_dynamic_iters:
        model.train()
        
        for i, data in enumerate(train_loader):
            x_t, a_t, x_tt = data
            x_t, a_t, x_tt = x_t.to(gpuid), a_t.to(gpuid), x_tt.to(gpuid)
            
            #new_task = btest.test(x_t, a_t)
            optimizer.zero_grad()
            output = model(x_t, a_t)
            loss = -mll(output, x_tt)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # SI update
            if hparams.model == "si":
                model.si_update()

            # Logging
            logger.train_step(loss)
            logger.validate(mll)

            it += 1
            if it >= hparams.train_dynamic_iters:
                break

def play_model(hparams):
    _, agent, checkpoint, _ = reload_model(hparams)

    # Reset seed
    reset_seed(hparams.seed)

    envs = CLEnvHandler(hparams.env, hparams.seed)
    for task_id in range(0, checkpoint['num_tasks_seen']):
        if hparams.model == "single":
            _, agent, _, _ = reload_model(hparams, task_id=task_id)

        env = envs.add_task(task_id, render=True)

        avg_rewards = []
        for _ in range(20):
            rewards = []
            x_t = env.reset()
            agent.reset()
            done = False
            while (not done):
                env.render()
                u_t = agent.act(x_t, task_id=task_id).cpu().numpy()
                x_tt, reward, done, info = env.step(u_t.reshape(env.action_space.shape))
                print(info)
                x_t = x_tt
                rewards.append(reward)
            eprew = np.sum(rewards)
            avg_rewards.append(eprew)
            print(f"Task {task_id + 1}, episode reward {eprew}, ep length {len(rewards)}")

        avg_reward = np.mean(avg_rewards)
        print(f"Average reward for task {task_id + 1} is {avg_reward}")

def run(hparams):

    # Reset seed
    reset_seed(hparams.seed)

    if hparams.resume:
        # Restore model and agent
        model, agent, checkpoint, collector = reload_model(hparams)
        # Boundary Test
        btest = BoundaryTest(model, hparams)

        # Restore logger
        logger = MonitorRL(hparams, agent, model, collector, btest)
        logger.load_stats(checkpoint)

        # Get num tasks we previously trained
        num_tasks_seen = checkpoint['num_tasks_seen']

    else:
        # Collect some random data
        collector = DataCollector(hparams)

        # Build model
        model = build_model(hparams)

        # Boundary Test
        btest = BoundaryTest(model, hparams)

        # RL Agent
        agent = MPC(hparams, model, collector=collector)

        # Monitor
        logger = MonitorRL(hparams, agent, model, collector, btest)

        # Start from scratch
        num_tasks_seen = 0

    # Convert to cuda
    model.to(hparams.gpuid)

    # Random Policy
    rand_pi = RandomAgent(hparams)   

    # Start learning in environment 
    envs = CLEnvHandler(hparams.env, hparams.seed)
    if hparams.resume:
        for tid in range(num_tasks_seen):
            envs.add_task(tid)

    for task_id in range(num_tasks_seen, hparams.num_tasks):
        # New Task with different friction
        env = envs.add_task(task_id, render=False)
        
        print(f"Collecting some random data first for task {task_id}")
        x_t = env.reset()
        for it in range(hparams.init_rand_steps):
            u = rand_pi.act(x_t)
            x_tt, _, done, _ = env.step(u.reshape(env.action_space.shape))
            collector.add(x_t, u, x_tt, task_id)
            x_t = x_tt

            if done:
                x_t = env.reset()

        # Augment Model
        trainer = augment_model_before(task_id, model, logger, collector, hparams)
        # Interact with the environment
        x_t = env.reset()
        agent.reset()
        for it in range(hparams.max_iteration):
            if it % hparams.dynamics_update_every == 0 and hparams.model != "gt":
                # Train Dynamics Model
                ts = time.time()
                train(task_id, model, trainer, logger, collector, btest, hparams)
                print('Training time', time.time() - ts)
            #env.render()
            u_t = agent.act(x_t, task_id=task_id).detach().cpu().numpy()
            x_tt, reward, done, info = env.step(u_t.reshape(env.action_space.shape))
                
            # Update the dataset of the env in which we're training 
            collector.add(x_t, u_t, x_tt, task_id)
            x_t = x_tt

            if done:
                x_t = env.reset()
                agent.reset()

            logger.env_step(x_tt, reward, done, info, task_id)

        # Update model after finishing task
        augment_model_after(task_id, model, logger, collector, hparams)

        # Save Model
        logger.save(task_id)

    envs.close()
    logger.writer.close()

def coreset(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "coreset"

    if play:
        play_model(hparams)
    else:
        run(hparams)

def ewc(env, seed=None, savepath=None, play=False):
    hparams = HP(env, seed, savepath)
    hparams.model = "ewc"

    # EWC beta
    hparams.ewc_beta = 10000
    hparams.ewc_online = False
    hparams.ewc_online_gamma = 1.
    hparams.empircal_fisher = True

    if play:
        play_model(hparams)
    else:
        run(hparams)

def si(env, seed=None, savepath=None, play=False):
    hparams = HP(env, seed, savepath)
    hparams.model = "si"

    # EWC beta
    hparams.si_c = 0.1 # si regularization strength
    hparams.si_epsilon = 1e-3 # si damping parameter

    if play:
        play_model(hparams)
    else:
        run(hparams)

def pnn(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "pnn"
    if play:
        play_model(hparams)
    else:
        run(hparams)


def finetune(env, seed=None, savepath=None, play=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath)
    hparams.model = "finetune"

    if play:
        play_model(hparams)
    else:
        run(hparams)

def multitask(env, seed=None, savepath=None, play=False):
    hparams = HP(env, seed, savepath)
    hparams.model = "multitask"
    hparams.mt_reinit = False

    if play:
        play_model(hparams)
    else:
        run(hparams)

def single(env, seed=None, savepath=None, play=False):
    hparmas = HP(env, seed, savepath)
    hparmas.model = "single"

    if play:
        play_model(hparmas)
    else:
        run(hparmas)


def gt(env, seed=None, savepath=None, play=False):
    hparams = HP(env, seed, savepath)
    hparams.model = "gt"
    hparams.control = "mpc-lqr"

    if play:
        play_model(hparams)
    else:
        run(hparams)

if __name__ == "__main__":
    import fire

    fire.Fire({
        'coreset': coreset,
        'pnn': pnn,
        'finetune': finetune,
        'single': single,
        "gt": gt,
        "ewc": ewc,
        "si": si,
        "multitask": multitask
    })
