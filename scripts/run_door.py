import subprocess
import multiprocessing
import os

seeds = [777, 855, 1045, 1046]
gpuids = [0, 0, 0, 0]
savepath = './runs/door'
env = 'door_pose'
robot = 'IIWA'
count = multiprocessing.cpu_count()


def work(t):
    if isinstance(t, tuple):
        cmd, gpuid = t
    else:
        cmd = t
        gpuid = 0

    os_env = os.environ.copy()
    os_env['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    return subprocess.run(cmd, shell=False, env=os_env)


pool = multiprocessing.Pool(processes=count)

models = [
    # 'single',
    # 'coreset',
    'hnet',
    # 'ewc',
    # 'si',
    # 'finetune',
    # 'multitask',
]

for model in models:
    cmd = [(['python3', 'main.py', model, env, robot, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
    pool.map(work, cmd)
