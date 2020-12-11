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
from hypercrl.tools import reset_seed, MonitorRL, MonitorHnet
from hypercrl.hnet_exp import TaskLoss
from hypercrl.model.mbrl import LogProbLoss, MSELoss

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

def work(arg):
    seed, models, envname, tasks, folder = arg

    stats = np.zeros((len(models), 2))
    reset_seed(seed)

    for (m, model) in enumerate(models):
        hpfile = osp.join(folder, f'TB{envname}_{model}_{seed}' , 'hparams.csv')
        hparams = read_hparams(folder, hpfile)
        
        if model in ['hnet', 'chunked_hnet', 'hnet_mt', 'hnet_si', 'hnet_ewc']:
            mnet, hnet, agent, checkpoint, collector = reload_model_hnet(hparams, task_id=0)
            # Restore Logger
            logger = MonitorHnet(hparams, agent, mnet, hnet, collector)
            #logger.load_stats(checkpoint)
            mll = TaskLoss(hparams, mnet)
        else:
            model, agent, checkpoint, collector = reload_model(hparams, task_id=0)
            # Restore logger
            logger = MonitorRL(hparams, agent, model, collector, None)
            #logger.load_stats(checkpoint)
            loss_fn = LogProbLoss if hparams.out_var else MSELoss
            mll = loss_fn(model, task_id=0, reg_lambda=hparams.reg_lambda)     
        # Train

        task_id = 0
        train_set, val_set = collector.get_dataset(task_id)
        loader = torch.utils.data.DataLoader(train_set, batch_size = hparams.bs,
                    num_workers=hparams.num_ds_worker)

        loss, diff = logger.validate_task(task_id, loader, mll, is_training=False)
        stats[m, 0] = diff.mean()

        # Val
        loader = torch.utils.data.DataLoader(val_set, batch_size = hparams.bs,
            num_workers=hparams.num_ds_worker)
        loss, diff = logger.validate_task(task_id, loader, mll, is_training=False)

        stats[m, 1] = diff.mean()

    return stats

def main(folder, envname, models, seeds, tasks):
    """
    Compute the reward after training each task
    """
    stats = np.zeros((len(models), len(seeds), 2))

    args = [(s, models, envname, tasks, folder) for s in seeds]
    with Pool(4) as pool:
        stats_list = pool.map(work, args)

    for s, stat in enumerate(stats_list):
        stats[:, s, :] = stat

    print(stats.mean(axis=1))

if __name__ == "__main__":
    folder = '/home/philiph/Documents/Continual-Learning/runs/lqr/door_pose/large_3x'
    env = 'door_pose'
    models = ['finetune']
    seeds = [777, 855, 1045, 1046]
    tasks = 5
    run_success_eval = False
    main(folder, env, models, seeds, tasks)
