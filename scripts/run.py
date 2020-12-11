import subprocess
import multiprocessing
import os

seeds = [777, 855, 1045, 1046]
gpuids = [0, 0, 0, 0]
savepath = './runs/lqr/pusher_new/hnet_replay'
# savepath='/home/philiph/scratch/door_pose/mt'
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

# cmd = [(['python3', 'main.py', 'single', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
# pool.map(work, cmd)
# cmd = [(['python3', 'main.py', 'chunked_hnet', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
# pool.map(work, cmd)
# cmd = [(['python3', 'main.py', 'coreset', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
# pool.map(work, cmd)
cmd = [(['python3', 'main.py', 'hnet_replay', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
pool.map(work, cmd)
# cmd = [(['python3', 'main.py', 'pnn', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
# pool.map(work, cmd)
# cmd = [(['python3', 'main.py', 'ewc', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
# pool.map(work, cmd)
# cmd = [(['python3', 'main.py', 'si', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
# pool.map(work, cmd)
# cmd = [(['python3', 'main.py', 'multitask', env, str(s), savepath], g) for s, g in zip(seeds, gpuids)]
# pool.map(work, cmd)


# Plotting command
models = ['hnet', 'hnet_replay']
num_tasks = 5
max_iteration = 4000
cmd = ['python3', 'scripts/plot_data.py',
    f'--base={savepath}',
    f'--env={env}',
    f'--models={models}',
    f'--tasks={num_tasks}',
    f'--seeds={seeds}',
    '--use_one_rl_fig=True',
    f'--max_iteration={max_iteration}',
    '--do_plot_train=True',
    '--do_plot_rl=True',
    '--plot_single=False',
    '--smoothing=False']
work(cmd)
