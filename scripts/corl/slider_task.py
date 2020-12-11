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

base = '/home/philiph/Documents/Continual-Learning/runs/lqr/pusher_slide/'
models = ['multitask', 'hnet', 'coreset', 'si', 'ewc', 'finetune']
models_label = ['multitask (oracle)', 'hnet', 'coreset', 'si', 'ewc', 'finetune'] 
seeds = [777, 855, 1045, 1046]
env = 'pusher_slide'
tasks = 5
train_len = 20
filename = osp.join(base, f'{env}_reward_after_each_task.csv')


# rewards = np.zeros((len(models), len(seeds), tasks, tasks))

# # Load Data
# for m, model in enumerate(models):
#     for s, seed in enumerate(seeds):
#         path = osp.join(base, f'RL{env}_{model}_{seed}.csv')
#         df = pd.read_csv(path)

#         for t in range(tasks):
#             df_t = df[df['task'] == t]
#             for j in range(t, tasks):
#                 row = df_t.iloc[(j - t + 1) * train_len - 1]
#                 rewards[m, s, t, j] = row['reward']

# # Save Data
# data_dict = {'Models': models}
# for t in range(tasks):
#     for j in range(t, tasks):
#         data_dict[f'Task {t}-{j}'] = rewards[:, :, t, j].mean(axis=1)
#         data_dict[f'Task {t}-{j}-std'] = rewards[:, :, t, j].std(axis=1)

# df = pd.DataFrame(data_dict)
# df.to_csv(filename)

# mu = rewards.mean(axis=1)
# std = rewards.std(axis=1)

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

# # Plot Data
# fig = plt.figure(figsize=(24, 9))
# nrow = 2
# ncol = (tasks + 1) // 2

# # Individual tasks
# for t in range(tasks):
#     ax = fig.add_subplot(nrow, ncol, t + 1)
#     xaxis = np.arange(t + 1, tasks + 1)

#     for m, model in enumerate(models):
#         y = mu[m, t, t:]
#         yerr = std[m, t, t:]
#         #plt.plot(xaxis, y, label=model)
#         #plt.scatter(xaxis, y)
#         plt.errorbar(xaxis, y, yerr, label=model, capsize=4)
    
#     plt.title( f'Task {t + 1}')
#     plt.xlabel('Task')
#     plt.ylabel('Reward')
#     plt.xlim(0.7, tasks + 0.3)
#     plt.ylim(0, 320)
#     ax.set_xticks(np.arange(1, tasks + 1).astype(int))
# plt.legend()

# # Final tasks
# ax = fig.add_subplot(nrow, ncol, tasks + 1)
# xaxis = np.arange(1, tasks + 1)
# for m, model in enumerate(models):
#     avg = []
#     avgerr = []
#     for j in range(tasks):
#         avg.append(mu[m, :j+1, j].mean())
#         avgerr.append(std[m, :j+1, j].std())
    
#     #plt.plot(xaxis, avg, label=model)
#     #plt.scatter(xaxis, avg)
#     plt.errorbar(xaxis, avg, avgerr, label=model, capsize=4)

# ax.set_xticks(np.arange(1, tasks + 1).astype(int))
# plt.title('Average')
# plt.xlabel('Task')
# plt.ylabel('Reward')
# plt.xlim(0.7, tasks + 0.3)
# plt.ylim(0, 320)

# plt.tight_layout()

# figname = osp.join(base, f'{env}_reward_after_each_task.pdf')
# fig.savefig(figname)

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