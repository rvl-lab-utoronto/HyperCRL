import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import json
import torch
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
from multiprocessing import Pool

from recordtype import recordtype
from hypercrl.model import reload_model, reload_model_hnet
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.tools import reset_seed, read_hparams

N_run_per_seed = 10

def isfloat(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def evaluate_reward(env, agent, task_id):
    ts = time.time()

    avg_rewards = []
    for _ in range(N_run_per_seed):
        x_t = env.reset()
        agent.reset()
        done = False
        eprew = 0
        while (not done):
            u_t = agent.act(x_t, task_id=task_id).cpu().numpy()
            x_tt, reward, done, info = env.step(u_t.reshape(env.action_space.shape))
            x_t = x_tt
            eprew += reward
        avg_rewards.append(eprew)

    avg_rewards =  np.mean(avg_rewards)
    print(f"Evaluating {N_run_per_seed} took {(time.time() - ts):.1f} s")
    return avg_rewards

def evaluate_success(env, agent, task_id):
    ts = time.time()

    success_rate = 0
    avgtime = []
    for _ in range(N_run_per_seed):
        x_t = env.reset()
        agent.reset()
        done = False
        t = 0
        while (not done):
            u_t = agent.act(x_t, task_id=task_id).cpu().numpy()
            x_tt, _, done, info = env.step(u_t.reshape(env.action_space.shape))
            x_t = x_tt
            success = info['success']
            t += 1
            if success:
                avgtime.append(t)
                break
        success_rate += int(success)
    
    avgtime = np.mean(avgtime)

    success_rate = success_rate / N_run_per_seed
    print(f"Evaluating {N_run_per_seed} took {(time.time() - ts):.1f} s")
    return success_rate, avgtime

def barplot(filename, stats, models, tasks):

    fig, ax = plt.subplots(figsize=(10, 7))

    labels = np.arange(1, tasks+1)
    x = np.arange(tasks)
    M = len(models)
    width = 0.7 / M
    offset = np.linspace(-(M-1)/2, (M-1)/2, M) * width

    rects_list = []
    for m, model in enumerate(models):
        mu = stats[m].mean(axis=0)
        std = stats[m].std(axis=0)

        rects = ax.bar(x + offset[m], mu, width, yerr=std, label=model, capsize=4)
        rects_list.append(rects)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test Reward')
    ax.set_xlabel('Tasks')
    ax.set_title('Pusher CL Rewards')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    for rects in rects_list:
        autolabel(rects)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def writecsv(filename, data_mat, style, tasks, models):
    with open(filename, 'w') as f:
        fieldnames = ['model']
        for t in range(tasks):
            fieldnames.append(str(t+1))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (m, model) in enumerate(models):
            row = {'model': model}
            for t in range(tasks):
                val = data_mat[m, :, t].mean()
                std = data_mat[m, :, t].std()
                if style == "%":
                    row[str(t+1)] = f'{val:.0%} ± {std:.0%}'
                else:
                    row[str(t+1)] = f'{val:.1f} ± {std:.1f}'
            writer.writerow(row)

def main_old(folder, envname, models, seeds, tasks, eval_single=True):
    """
    Compute the average final reward on given env with given models
    Optionally:
        1. Also compute the forgeting
        2. Also compute the forward transfer
    """
    stats = np.zeros((len(models), len(seeds), tasks))
    single_stats = np.zeros((len(seeds), tasks))

    for (s, seed) in enumerate(seeds):
        reset_seed(seed)
        envs = CLEnvHandler(envname, seed)
        for t in range(tasks):
            envs.add_task(t, render=False)

        for (m, model) in enumerate(models):
            hpfile = osp.join(folder, f'TB{envname}_{model}_{seed}' , 'hparams.csv')
            hparams = read_hparams(folder, hpfile)

            # Reload the final model
            if model in ['hnet', 'chunked_hnet', 'hnet_mt', 'hnet_si', 'hnet_ewc']:
                _, _, agent, _, _ = reload_model_hnet(hparams)
            else:
                _, agent, _, _ = reload_model(hparams)

            for t in range(tasks):
                env = envs.get_env(t)
                rew = evaluate_reward(env, agent, t)
                stats[m, s, t] = rew
        
        if eval_single:
            hpfile = osp.join(folder, f'TB{envname}_single_{seed}' , 'hparams.csv')
            hparams = read_hparams(folder, hpfile)

            for t in range(tasks):
                # Reload the model
                _, agent, _, _ = reload_model(hparams, task_id=t)

                env = envs.get_env(t)
                rew = evaluate_reward(env, agent, t)
                single_stats[s, t] = rew
    
    # Create bar charts of final reward on different methods 
    filename = osp.join(folder, f'{envname}_final_reward.png')
    barplot(filename, stats, models, tasks)

    # Save results
    filename = osp.join(folder, f'{envname}_final_reward.csv')
    writecsv(filename, stats, 'f', tasks, models)

    if eval_single:
        # forgetting
        og_stats = np.zeros((len(models), len(seeds), tasks))

        for (s, seed) in enumerate(seeds):
            envs = CLEnvHandler(envname, seed)
            for t in range(tasks):
                envs.add_task(t, render=False)

            for (m, model) in enumerate(models):
                hpfile = osp.join(folder, f'TB{envname}_{model}_{seed}' , 'hparams.csv')
                hparams = read_hparams(folder, hpfile)

                for t in range(tasks):
                    # Reload the model at t
                    if model in ['hnet', 'chunked_hnet']:
                        _, _, agent, _, _ = reload_model_hnet(hparams, task_id=t)
                    else:
                        _, agent, _, _ = reload_model(hparams, task_id=t)
                    
                    env = envs.get_env(t)
                    rew = evaluate_reward(env, agent, t)
                    og_stats[m, s, t] = rew

        forgetting = stats / og_stats
        filename = osp.join(folder, f'{envname}_forgetting.csv')
        writecsv(filename, forgetting, '%', tasks, models)

        # Positive Transfer
        filename = osp.join(folder, f'{envname}_forward_transfer.csv')
        pos_trans = (og_stats / single_stats)
        writecsv(filename, pos_trans, '%', tasks, models) 

def work(arg):
    seed, models, envname, tasks, folder, run_s = arg

    stats = np.zeros((len(models), tasks, tasks))
    reset_seed(seed)
    envs = CLEnvHandler(envname, seed)
    for t in range(tasks):
        envs.add_task(t, render=False)

    for (m, model) in enumerate(models):
        hpfile = osp.join(folder, f'TB{envname}_{model}_{seed}' , 'hparams.csv')
        hparams = read_hparams(folder, hpfile)

        for j in range(tasks):
            # Reload the model trained after task j
            if model in ['hnet', 'chunked_hnet', 'hnet_mt', 'hnet_si', 'hnet_ewc']:
                _, _, agent, _, _ = reload_model_hnet(hparams, task_id=j)
            else:
                _, agent, _, _ = reload_model(hparams, task_id=j)
            for t in range(j + 1):
                env = envs.get_env(t)
                if model == "hnet":
                    agent.cache_hnet(t)
                if run_s:
                    rew, avgtime = evaluate_success(env, agent, t)
                    print(f"{model}, Task{t}-{j}: {rew} / {avgtime}")
                else:
                    rew = evaluate_reward(env, agent, t)
                    print(f"{model}, Task{t}-{j}: {rew}")
                stats[m, t, j] = rew
    return stats

def main(folder, envname, models, seeds, tasks, run_s):
    """
    Compute the reward after training each task
    """
    stats = np.zeros((len(models), len(seeds), tasks, tasks))

    args = [(s, models, envname, tasks, folder, run_s) for s in seeds]
    with Pool(4) as pool:
        stats_list = pool.map(work, args)

    for s, stat in enumerate(stats_list):
        stats[:, s, :, :] = stat

    # Save Data
    data_dict = {'Models': models}
    for t in range(tasks):
        for j in range(t, tasks):
            data_dict[f'Task {t}-{j}'] = stats[:, :, t, j].mean(axis=1)
            data_dict[f'Task {t}-{j}-std'] = stats[:, :, t, j].std(axis=1)

    df = pd.DataFrame(data_dict)
    if run_s:
        filename = osp.join(folder, f'{envname}_{models[0]}_success_after_each_task.csv')
    else:
        filename = osp.join(folder, f'{envname}_{models[0]}_reward_after_each_task.csv')
    df.to_csv(filename)

def work_single(arg):
    seed, envname, tasks, folder, run_s = arg

    stats = np.zeros(tasks)
    reset_seed(seed)
    envs = CLEnvHandler(envname, seed)

    # Load Hparam
    hpfile = osp.join(folder, f'TB{envname}_single_{seed}' , 'hparams.csv')
    hparams = read_hparams(folder, hpfile)

    for t in range(tasks):
        env = envs.add_task(t, render=False)
        # Reload the model trained after task j
        _, agent, _, _ = reload_model(hparams, task_id=t)
        if run_s:
            rew, avgtime  = evaluate_success(env, agent, t)
            print(f"single {rew}/{avgtime}")
        else:
            rew = evaluate_reward(env, agent, t)
        stats[t] = rew

    return stats

def main_single(folder, envname, seeds, tasks, run_s):
    stats = np.zeros((len(seeds), tasks))

    args = [(s, envname, tasks, folder, run_s) for s in seeds]
    with Pool(4) as pool:
        stats_list = pool.map(work_single, args)

    for s, stat in enumerate(stats_list):
        stats[s, :] = stat

    data = {}
    for t in range(tasks):
        data[f'Task {t}'] = [stats[:, t].mean()]
        data[f'Task {t}-std'] = [stats[:, t].std()]

    df = pd.DataFrame(data)

    if run_s:
        filename = osp.join(folder, f'{envname}_single_success_rate.csv')
    else:
        filename = osp.join(folder, f'{envname}_single_reward.csv')
    df.to_csv(filename)


if __name__ == "__main__":
    folder = '/home/philiph/Documents/Continual-Learning/runs/lqr/pushe_new/penalty'
    env = 'pusher'
    models = ['hnet_mt']
    seeds = [777, 855, 1045, 1046]
    tasks = 5
    run_success_eval = False
    #main_old(folder, env, models, seeds, tasks, eval_single=True)
    main(folder, env, models, seeds, tasks, run_success_eval)
    #main_single(folder, env, seeds, tasks, run_success_eval)
