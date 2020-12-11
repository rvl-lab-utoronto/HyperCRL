import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os.path as osp

sns.set_style("whitegrid", rc={'grid.linestyle': '--'})
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2, "lines.markersize": 5})
sns.set_palette("colorblind", n_colors=8, desat=1)
# sns.set('talk', 'whitegrid', 'dark', font_scale=2.2,
#         rc={"lines.linewidth": 4, 'grid.linestyle': '--'})


base = '/home/philiph/Documents/Continual-Learning/runs/lqr/door_pose'
models = ['multitask', 'hnet', 'coreset', 'si', 'ewc', 'finetune']
models_label = ['multitask (oracle)', 'hnet', 'coreset', 'si', 'ewc', 'finetune']
seeds = [777, 855, 1045, 1046]
env = 'door_pose'
tasks = 5
filename = osp.join(base, f'{env}_reward_after_each_task.csv')

#normalized max taken from average performance of single task
normalize_max = np.array([2029.74748184584, 2609.68930587166, 5904.12831782415, 1154.12178509701, 3996.37851282015])
#normalized min taken from minimum performance of single task 
normalize_min = np.array([-634.5806438, -1550.238556 , -1339.698824 , -1678.947256 ,
       -1754.515303])

# ################################################################################
### Load 10 epsidoe-aveaged data

mu = np.zeros((len(models), tasks, tasks))
std = np.zeros((len(models), tasks, tasks))

df = pd.read_csv(filename)
for m, model in enumerate(models):
    row = df[df["Models"] == model].iloc[0]
    for t in range(tasks):
        for j in range(t, tasks):
            mu[m, t, j] = row[f'Task {t}-{j}']
            std[m, t, j] = row[f'Task {t}-{j}-std']

# Normalize
mu = (mu - normalize_min[:, None]) / (normalize_max[:, None] - normalize_min[:, None])
std = (std - normalize_min[:, None]) / (normalize_max[:, None] - normalize_min[:, None])

def div_cal(x, y, dx, dy):
    z = x / y
    dz = z * (((dx / x)**2 + (dy/y) ** 2) ** 0.5)

    return z, dz

# Print Forgetting Measure
print("Forgetting")
for m, model in enumerate(models):
    print(model, "=================")
    avg = np.zeros((tasks, 2))
    for t in range(tasks - 1):
        forgetting, error = div_cal(mu[m, t, -1], mu[m, t, t], std[m, t, -1], std[m, t, t])
        avg[t, 0] = forgetting
        avg[t, 1] = error
        print("Task", t+1, forgetting, error)
    forgetting = np.mean(avg[:-1, 0])
    error = (((avg[:-1, 1] ** 2).sum())**0.5) / (tasks-1)
    print("Average: ", forgetting, error)

### Positive Transfer
s_mu = np.zeros(tasks)
s_std = np.zeros(tasks)
# Load Data 
df = pd.read_csv(osp.join(base, f'{env}_single_reward.csv'))
for t in range(tasks):
    s_mu[t] = df.iloc[0][f'Task {t}']
    s_std[t] = df.iloc[0][f'Task {t}-std']

# Normalize
s_mu = (s_mu - normalize_min) / (normalize_max - normalize_min)
s_std = (s_std - normalize_min) / (normalize_max - normalize_min)

print("Forward Transfer")
for m, model in enumerate(models):
    print (model, "==========")
    avg = np.zeros((tasks, 2))
    for t in range(1, tasks):
        pos_trans, error = div_cal(mu[m, t, t], s_mu[t], std[m, t, t], s_std[t])
        print("Task", t+1, pos_trans, error)
        avg[t, 0] = pos_trans
        avg[t, 1] = error

    forward = np.mean(avg[1:, 0])
    error = (((avg[1:, 1] ** 2).sum())**0.5) / (tasks-1)
    print("Average", forward, error)
