import csv
import os.path as osp
import numpy as np
import math
import pandas as pd
import fire
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_training(base, env, models, tasks, seeds, use_one_rl_fig, max_iteration,
        task_description, plot_single, smoothing):
    stats = {}
    times = {}

    for t in range(tasks):
        stats[t] = [[[] for s in range(len(seeds))] for m in range(len(models))]
        times[t] = []

    record_time = True
    for m, model in enumerate(models):
        for s, seed in enumerate(seeds):
            filename = osp.join(base, f'{env}_{model}_{seed}.csv')
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                
                arr = [[] for t in range(tasks)]

                for row in reader:
                    task = int(row['task'])
                    diff = float(row['diff'])
                    
                    arr[task].append(diff)

                    if record_time:
                        times[task].append(int(row['time']))
                
                for t in range(tasks):
                    stats[t][m][s] = np.array(arr[t])
            record_time = False

    if plot_single:
        model = 'single'
        bounds = {}
        singles = {}
        for t in range(tasks):
            singles[t] = [[] for s in range(len(seeds))]
            bounds[t] = [999 for s in range(len(seeds))]

        for s, seed in enumerate(seeds):
            filename = osp.join(base, f'{env}_{model}_{seed}.csv')
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
            
                for row in reader:
                    task = int(row['task'])
                    diff = float(row['diff'])

                    bounds[task][s] = min(bounds[task][s], diff)
                    singles[task][s].append(diff)

    for t in range(tasks):
        arr = np.array(stats[t])

        time = times[t]
        fig = plt.figure(figsize=(16, 9))
        if plot_single:
            bound = np.array(bounds[t]).mean()
            single_diff = np.array(singles[t]).mean(axis=0)
            std = np.array(singles[t]).std(axis=0)
            # plt.plot(time, bound * np.ones(len(time)), 'k-', label='single task')
            plt.plot(time[:len(single_diff)], single_diff, '-', label='single task')
            plt.fill_between(time[:len(single_diff)], single_diff-std, single_diff+std, alpha=0.5)

        for m in range(arr.shape[0]):
            mu=arr[m, :].mean(0)
            mu = savgol_filter(mu, 5, 3) if smoothing else mu
            std=arr[m, :].std(0)
            plt.plot(time, mu, '--', label = models[m])
            plt.fill_between(time, mu-std, mu+std, alpha=0.5)
        
        plt.ylabel('l1 diff')
        plt.xlabel('training steps')
        plt.title(f'{env}_task{t}')
        
        plt.xlim([-1000, max(time) + 1000])
        #plt.ylim([0, 0.1])
        plt.legend()
        filename = osp.join(base, f'{env}_task{t}.png')
        plt.savefig(filename)
        plt.close()

def plot_rl(base, env, models, tasks, seeds, use_one_rl_fig, max_iteration,
        task_description, plot_single, smoothing):
    ################################### RL #####################################
    rewards, diffs = {}, {}
    times = {}

    for t in range(tasks):
        rewards[t] = [[[] for s in range(len(seeds))] for m in range(len(models))]
        diffs[t] = [[[] for s in range(len(seeds))] for m in range(len(models))]
        times[t] = []

    record_time = True
    for m, model in enumerate(models):
        for s, seed in enumerate(seeds):
            filename = osp.join(base, f'RL{env}_{model}_{seed}.csv')
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    task = int(row['task'])
                    reward = float(row['reward'])
                    diff = float(row['on_policy_diff'])
                    
                    rewards[task][m][s].append(reward)
                    diffs[task][m][s].append(diff)

                    if record_time:
                        times[task].append(int(row['envstep']))

            record_time = False

    single_rewards, single_diffs = {}, {}
    if plot_single:
        # Single Model Baseline
        single_times = {}

        for t in range(tasks):
            single_rewards[t] = [[] for s in range(len(seeds))]
            single_diffs[t] = [[] for s in range(len(seeds))]
            single_times[t] = []

        record_time = True
        for s, seed in enumerate(seeds):
            filename = osp.join(base, f'RL{env}_single_{seed}.csv')
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    task = int(row['task'])
                    reward = float(row['reward'])
                    diff = float(row['on_policy_diff'])

                    single_rewards[task][s].append(reward)
                    single_diffs[task][s].append(diff)
                    if record_time:
                        single_times[task].append(int(row['envstep']))

            record_time = False

    def plot_rl_in_ax(t, ydata, ydata_single, ylabel, ylim, use_one_rl_fig):
        # Data
        arr = np.array(ydata[t])
        time = times[t]

        # Single task baseline
        if plot_single:
            single = np.array(ydata_single[t])
            single_time = single_times[t]
            mu = single.mean(0)
            std = single.std(0)
            plt.plot(single_time, mu, '--', label = "single_task")
            plt.fill_between(single_time, mu-std, mu+std, alpha=0.2)

        # Vertical line indicating task switch
        st = time[0] // max_iteration
        xswitch = np.arange(st, tasks) * max_iteration
        for xc in xswitch:
            plt.axvline(x=xc, color='gray', linestyle='--')

        # Models
        for m in range(arr.shape[0]):
            mu = arr[m, :].mean(0)
            mu = savgol_filter(mu, 5, 3) if smoothing else mu
            std=arr[m, :].std(0)
            plt.plot(time, mu, '--', label = models[m])
            plt.fill_between(time, mu-std, mu+std, alpha=0.2)
        
        plt.ylabel(ylabel)
        plt.xlabel('env steps')
        plt.title(f'{env}_{task_description[t]}')

        if use_one_rl_fig:
            if len(time) != 0:
                plt.xlim([-10, max(time) + 10])
        if ylim is not None:
            plt.ylim(ylim)

    def plot_one_data(data, single_data, label, ylim):
        if use_one_rl_fig:
            if tasks <= 5:
                fig = plt.figure(figsize=(16, 9))
                num_row, num_col = tasks, 1
            else:
                fig = plt.figure(figsize=(24, 9))
                num_row, num_col = int(tasks/2+0.5), 2
            for t in range(tasks):
                fig.add_subplot(num_row, num_col, t + 1)
                plot_rl_in_ax(t, data, single_data, label, ylim, use_one_rl_fig)
            
            plt.legend()
            plt.tight_layout()
            filename = osp.join(base, f'{env}_{label}.png')
            plt.savefig(filename)
            plt.close()

        else:
            for t in range(tasks):
                fig = plt.figure(figsize=(16, 9))
                plot_rl_in_ax(t, data, single_data, label, ylim, use_one_rl_fig)

                plt.legend()
                filename = osp.join(base, f'{env}_task{t}_{label}.png')
                plt.savefig(filename)
                plt.close()

    plot_one_data(rewards, single_rewards, 'reward', None)
    #plot_one_data(diffs, single_diffs, 'onpoli_diff', [0, 0.01])

def plot_debug_dynamic(base, env, models, tasks, seeds, max_iteration, task_description, smoothing):
    manual_diffs = [[[] for s in range(len(seeds))] for m in range(len(models))]
    rand_diffs = [[[] for s in range(len(seeds))] for m in range(len(models))]
    times = []

    record_time = True
    for m, model in enumerate(models):
        for s, seed in enumerate(seeds):
            filename = osp.join(base, f'Debug{env}_{model}_{seed}.csv')
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    manual_diff = float(row['p_manual_diff'])
                    rand_diff = float(row['p_rand_diff'])
                    manual_diffs[m][s].append(manual_diff)
                    rand_diffs[m][s].append(rand_diff)

                    if record_time:
                        times.append(int(row['time']))

            record_time = False

    manual_diffs = np.array(manual_diffs)
    rand_diffs = np.array(rand_diffs)

    fig = plt.figure(figsize=(16, 9))

    # Vertical line indicating task switch
    xswitch = np.arange(1, tasks) * 40000
    for xc in xswitch:
        plt.axvline(x=xc, color='gray', linestyle='--')

    for m in range(len(models)):
        # mu = manual_diffs[m, :].mean(0)
        # mu = savgol_filter(mu, 5, 3) if smoothing else mu
        # std = manual_diffs[m, :].std(0)
        # plt.plot(times, mu, '--', label = models[m] + " manual")
        # plt.fill_between(times, mu-std, mu+std, alpha=0.5)

        mu = rand_diffs[m, :].mean(0)
        mu = savgol_filter(mu, 5, 3) if smoothing else mu
        std = rand_diffs[m, :].std(0)
        plt.plot(times, mu, '-.', label= models[m] + " rand")
        plt.fill_between(times, mu-std, mu+std, alpha=0.5)

    plt.ylabel('l1 prediction error')
    plt.xlabel('training steps')
    plt.title(f'{env}_task 0')
        
    plt.xlim([-1000, max(times) + 1000])
    plt.ylim([0, 0.01])
    plt.legend()
    filename = osp.join(base, f'Debug{env}_task_0.png')
    plt.savefig(filename)
    plt.close()


def plot_data(base='./runs/lqr/lqr10',
    env="lqr10",
    models=["hnet", "baseline"],
    tasks=4,
    seeds=[1045, 1046, 2000, 2344, 777, 855],
    use_one_rl_fig=True,
    max_iteration=10000,
    do_plot_train=True,
    do_plot_rl=True,
    plot_single=True,
    smoothing=True):

    tasks = int(tasks)

    if env.startswith("lqr"):
        frictions = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        task_description = [f"friction={f}" for f in frictions]
    elif env.startswith("pendulum2"):
        task_description = [f"g={g}" for g in [10, 0, 2, 8, 12, 5, 4, 14, 9, 7, 30]]
    elif env.startswith("half_cheetah_body"):
        task_description = ["normal", "big torso", "big thigh", "big leg", "big foot"]
    elif env.startswith("half_cheetah"):
        task_description = [f"task={i}" for i in range(10)]
    elif env.startswith("inverted_pendulum"):
        task_description = [f"g={g}" for g in [10, 0, 2, 8, 12, 5, 4, 14, 9, 7, 30]]
    elif env == "cartpole":
        task_description = ["l=0.6", "l=0.8", "l=0.4", "l=1.0", "l=1.2", "l=0.7",
                            "l=0.5", "l=0.9", 'l=1.1', 'l=1.3']
    elif env == "cartpole_bin":
        task_description = ["-2.5-2.5", "-7.5~-2.5", "2.5~7.5"]
    elif env == "reacher":
        task_description = ['l=0.10, 0.10', 'l=0.09, 0.09', 'l=0.11, 0.11', 'l=0.08, 0.08',
                            'l=0.12, 0.12', 'l=0.07, 0.09', 'l=0.13, 0.13', 'l=0.09, 0.07',
                            'l=0.12, 0.10', 'l=0.08, 0.10']
    elif env == "hopper_body":
        task_description = ['normal', 'big torso', 'big thigh']
    elif env == "pusher":
        task_description = ["1x", "5x", "0.2x", "2x", "0.5x", "10x", "0.1x", "0.3x", "3.3x", "1x"]
    elif env == "door":
        task_description = ["pull"]
    elif env == "door_pose":
        task_description = ["lever", "round", "pull", "round_cc", "pull_cc"]
    elif env == "pusher_slide":
        task_description = ['friction=0.001', '0.0005', '0.002', '0.0026', '0.005']

    if do_plot_train:
        plot_training(base, env, models, tasks, seeds, use_one_rl_fig, max_iteration,
            task_description, plot_single, smoothing)
    if do_plot_rl:
        plot_rl(base, env, models, tasks, seeds, use_one_rl_fig, max_iteration,
            task_description, plot_single, smoothing)

    # plot_debug_dynamic(base, env, models, tasks, seeds, max_iteration,
    #         task_description, smoothing)

if __name__ == "__main__":
    fire.Fire(plot_data)