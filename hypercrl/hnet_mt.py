import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from hypercrl.dataset.datautil import DataCollector
from hypercrl.tools import MonitorHnet, HP, Hparams
from hypercrl.hnet_exp import TaskLossMT, augment_model, augment_model_after, train
from hypercrl.model import build_model_hnet, reload_model_hnet
from hypercrl.control import MPC
from hypercrl.envs.cl_env import CLEnvHandler


def hnet_mt_sd(seed):
    """
    hnet mt same data
    """

    env = "door_pose"
    base = "./runs/lqr/door_pose/"
    hparams = HP(env, seed, base)
    hparams.model = "hnet_mt_sd"

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0
    hparams.plastic_prev_tembs = True

    # MT specific hparmas
    hparams.train_dynamic_iters = 60000
    hparams.eval_env_run_every = 1000
    hparams.run_eval_env_eps = 10


    # Build Model
    mnet, hnet = build_model_hnet(hparams)

    # Restore Data
    collector = MonitorHnet.resume_from_disk(hparams)

    # RL Agent
    agent = MPC(hparams, mnet, collector=collector, hnet=hnet)

    # Monitor
    logger = MonitorHnet(hparams, agent, mnet, hnet, collector)

    # Convert to cuda
    mnet.to(hparams.gpuid)
    hnet.to(hparams.gpuid)
    
    # Train Model for each task
    for task_id in range(hparams.num_tasks):

        # Augment Model, instantiate optimizers/regularizer targets
        trainer_misc = augment_model(task_id, mnet, hnet, collector, hparams)

        # Train Dynamics Model
        ts = time.time()
        train(task_id, mnet, hnet, trainer_misc, logger, collector, hparams)
        print("Training time", time.time() - ts)
        
        logger.env_step(None, None, False, None, task_id)
        augment_model_after(task_id, mnet, hnet, hparams, collector)

        # Save Model
        logger.save(task_id)

    logger.writer.close()