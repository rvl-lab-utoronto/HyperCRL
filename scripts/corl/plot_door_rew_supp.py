import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import csv
import os.path as osp
from matplotlib.ticker import FormatStrFormatter

sns.set_style("white", rc={'grid.linestyle': '--'})
sns.set_context("paper", font_scale=2.6, rc={"lines.linewidth": 3.5, "lines.markersize": 5,
        "legend.fontsize": 27, "axes.labelsize": 27})
sns.set_palette("bright", n_colors=8, desat=1)
# sns.set('talk', 'whitegrid', 'dark', font_scale=2.2,
#         rc={"lines.linewidth": 4, 'grid.linestyle': '--'})

base = '/home/philiph/Documents/Continual-Learning/runs/lqr/door_pose'
models = ['hnet', 'multitask', 'si', 'ewc', 'coreset', 'finetune']
models_label = ['HyperCRL (original)', 'multitask (oracle)', 'SI', 'EWC', 'coreset', 'finetuning']
seeds = [777, 855, 1045, 1046]
env = 'door_pose'
tasks = 5
num_per_task = 60
max_iteration = 60000

#normalized max taken from average performance of single task
normalize_max = np.array([2029.74748184584, 2609.68930587166, 5904.12831782415, 1154.12178509701, 3996.37851282015])
#normalized min taken from minimum performance of single task 
normalize_min = np.array([-634.5806438, -1550.238556 , -1339.698824 , -1678.947256 ,
       -1754.515303])

def normalize(reward, t):
    norm_r = (reward - normalize_min[t]) / (normalize_max[t] - normalize_min[t])
    return norm_r

single_rewards, single_diffs = {}, {}
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

            single_rewards[task][s].append(reward)
            if record_time:
                single_times[task].append(int(row['envstep']))

    record_time = False

################################### RL #####################################
rewards = {}
times = {}

for t in range(tasks):
    rewards[t] = [[[] for s in range(len(seeds))] for m in range(len(models))]
    times[t] = []

record_time = True
for m, model in enumerate(models):
    for s, seed in enumerate(seeds):
        if model != "hnet":
            filename = osp.join(base, 'large_3x', f'RL{env}_{model}_{seed}.csv')
        else:
            filename = osp.join(base, f'RL{env}_{model}_{seed}.csv')
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                task = int(row['task'])
                reward = float(row['reward'])
                reward = normalize(reward, task)
                
                rewards[task][m][s].append(reward)

                if record_time:
                    times[task].append(int(row['envstep']))

        record_time = False

######################### Plotting #############################################

fig = plt.figure(figsize=(24, 12))
num_row, num_col = 3, 2
for t in range(tasks):
    fig.add_subplot(num_row, num_col, t + 1)
    # Data
    arr = np.array(rewards[t])
    time = times[t]

    # Vertical line indicating task switch
    st = time[0] // max_iteration
    xswitch = np.arange(1, tasks) * max_iteration
    for xc in xswitch:
        plt.axvline(x=xc, color='gray', linestyle='--', linewidth=2)

    # Models
    for m in range(arr.shape[0]):
        mu = arr[m, :].mean(0) 
        std = arr[m, :].std(0)
        plt.plot(time, mu, '-')
        plt.fill_between(time, mu-std, mu+std, alpha=0.2)

    # # Single task baseline
    # single = np.array(single_rewards[t])
    # single_time = single_times[t]
    # mu = single.mean(0)
    # std = single.std(0)
    # plt.plot(single_time, mu, '--', label = "single_task")
    # plt.fill_between(single_time, mu-std, mu+std, alpha=0.2)

    if t == 4:
        plt.xlabel('Environment Steps')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='major',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            direction='out',  # direction of ticks
            labelbottom=True) # labels along the bottom edge are off
    else:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='major',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            direction='out',  # direction of ticks
            labelbottom=False) # labels along the bottom edge are off
    #plt.title(f'Task {t + 1}')
    plt.xlim([-10, max(time) + 10])

    # ylabel
    plt.ylabel(f"Task {t+1}")
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        left=True,      # ticks along the left edge are off 
        direction='out',  # direction of ticks
        labelleft=True) # labels along the left edge are off
    if t == 0:
        plt.ylim([-0.4, 1.7])
    #plt.ylim([-0.4, 2.3])
    if t == 3:
        fig.axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Plotting Average Reward
fig.add_subplot(num_row, num_col, tasks + 1)

# Vertical line indicating task switch
st = time[0] // max_iteration
xswitch = np.arange(1, tasks) * max_iteration
for xc in xswitch:
    plt.axvline(x=xc, color='gray', linestyle='--', linewidth=2)

for m, model in enumerate(models):
    data =  np.zeros((len(seeds), num_per_task * tasks))
    for t in range(tasks):
        seg_st = t * num_per_task
        seg_ed = (t + 1) * num_per_task
        for j in range(0, t + 1):
            arr = np.array(rewards[j][m])
            st = (t -j) * num_per_task
            ed = (t - j+ 1) * num_per_task
            data[:, seg_st:seg_ed] += arr[:, st:ed]

        data[:, seg_st:seg_ed] /= (t+1)

    time = times[0]
    mu = data.mean(0)
    std = data.std(0)
    plt.plot(time, mu, '-', label = models_label[m])
    plt.fill_between(time, mu-std, mu+std, alpha=0.2)

#plt.title('Average')
plt.ylabel('Average')

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    left=True,      # ticks along the left edge are off 
    direction='out',  # direction of ticks
    labelleft=True) # labels along the left edge are off

# X-axis label and tick
plt.xlim([-10, max(time) + 10])
#plt.xlabel('Env Steps')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    direction='out',  # direction of ticks
    labelbottom=True) # labels along the bottom edge are off

#plt.ylim([-0.4, 2.3])

fig.text(0.003, 0.5, 'Normalized Reward', va='center', rotation='vertical')
fig.legend(bbox_to_anchor=(0.5, 1.0), loc="upper center", ncol=6)
plt.subplots_adjust(wspace=0.12, hspace=0.1, left=0.07, right=0.985, top=0.93, bottom=0.07)

#plt.tight_layout()
filename = osp.join(base, f'{env}_reward_supp.pdf')
plt.savefig(filename)
plt.close()
