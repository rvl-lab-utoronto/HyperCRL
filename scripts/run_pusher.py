import subprocess
import multiprocessing
import os

seeds = [777, 855, 1045, 1046]
gpuids = [0, 0, 0, 0]
savepath = './runs/pusher'
env='pusher'
count = 4

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

cmd = [(['python3', 'main.py', 'single', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)
cmd = [(['python3', 'main.py', 'coreset', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)
cmd = [(['python3', 'main.py', 'hnet', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)
cmd = [(['python3', 'main.py', 'ewc', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)
cmd = [(['python3', 'main.py', 'si', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)
cmd = [(['python3', 'main.py', 'finetune', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)
cmd = [(['python3', 'main.py', 'multitask', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)


