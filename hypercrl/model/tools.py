import torch
from torch.nn import functional as F
import numpy as np

import os
from .mbrl import Baseline, PNN, MTBaseline
from .regularizer import BaselineReg

from hypercrl.tools import MonitorRL, MonitorHnet, str_to_act, Hparams
from hypercrl.control import MPC

from hypercrl.hypercl import HyperNetwork, MLP, ChunkedHyperNetworkHandler, MainNetInterface
from hypercrl.hypercl.utils import ewc_regularizer as ewc
from hypercrl.hypercl.utils import si_regularizer as si

def get_recon_loss(decoder_out, X, dist, reduction='sum', weight=1):
    if dist == "bernoulli":
        X_sample = decoder_out
        loss = F.binary_cross_entropy(X_sample, X, reduction='sum') / X.size(0)
    elif dist == "gaussian":
        mu, logscale = decoder_out
        err = (X - mu) * torch.exp(-logscale)
        loss = 0.5 * err * err + logscale + 0.5 * np.log(np.pi * 2)

        if reduction=='sum':
            loss = torch.sum(loss) * weight
        elif reduction=='batch_mean':
            loss = torch.sum(loss) / X.size(0) * weight
        elif reduction=='mean':
            loss = torch.mean(loss) * weight
    return loss

def get_dual_loss(T_q, T_p, reduction='mean'):
    loss_q = F.binary_cross_entropy_with_logits(T_q, torch.ones_like(T_q), reduction=reduction)
    loss_p = F.binary_cross_entropy_with_logits(T_p, torch.zeros_like(T_p), reduction=reduction)

    loss = loss_p + loss_q
    return loss

def log_normal(x, mu, logs, axis=-1):
    return -0.5 * torch.sum((x - mu).pow(2) * (-logs).exp() + logs + np.log(2 * np.pi), axis)

#######################  Model UTIL ####################################

def build_model(hparams):
    # Build Dynamics Model and loss

    # Specific environment model converts angle to [cos, sin]
    if hparams.env.startswith("inverted_pendulum") or hparams.env.startswith('cartpole') \
        or hparams.env == "door":
        import copy
        hparams = copy.deepcopy(hparams)
        hparams.state_dim = hparams.state_dim + 1
    elif hparams.env == "door_pose":
        import copy
        hparams = copy.deepcopy(hparams)
        hparams.state_dim = hparams.state_dim + 2

    if hparams.model in ["coreset", "finetune", "single", "gt"]:
        # Baseline
        model = Baseline(hparams)
    elif hparams.model == "multitask":
        model = MTBaseline(hparams)
    elif hparams.model == "pnn":
        # PNN
        model = PNN(hparams)
    elif hparams.model == "ewc":
        model = BaselineReg(hparams)
    elif hparams.model == "si":
        model = BaselineReg(hparams)
        model.si_register_buffer()

    return model

def reload_model(hparams, task_id=None):
    model = build_model(hparams)
    # Restore Data
    collector = MonitorRL.resume_from_disk(hparams)
    agent = MPC(hparams, model, collector=collector)

    # Load Checkpoint
    if task_id is None:
        model_path = os.path.join(hparams.save_folder,
            f'TB{hparams.env}_{hparams.model}_{hparams.seed}', 'model.pt')
    else:
        model_path = os.path.join(hparams.save_folder,
            f'TB{hparams.env}_{hparams.model}_{hparams.seed}', f'model_{task_id}.pt')     
    checkpoint = torch.load(model_path, map_location=hparams.gpuid)
    print("Checkpoint Loaded")

    # Add task to the model
    for task_id in range(checkpoint['num_tasks_seen']):
        model.add_weights(task_id)

    # Register SI/EWC buffer
    if hparams.model == "ewc":
        model.ewc_register_buffer(checkpoint['num_tasks_seen'])
    elif hparams.model == "coreset":
        for task_id in range(checkpoint['num_tasks_seen']):
            z_x = checkpoint['model_state_dict'][f'z_x{task_id}']
            z_a = checkpoint['model_state_dict'][f'z_a{task_id}']
            z_y = checkpoint['model_state_dict'][f'z_y{task_id}']
            model.add_inducing_points([z_x, z_a, z_y], task_id)
    elif hparams.model == "multitask":
        model.add_trainset(collector, checkpoint['num_tasks_seen'] - 1)

    if hparams.model == "ewc":
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load model to gpu
    model.to(hparams.gpuid)
    print("Model restored")

    return model, agent, checkpoint, collector

####################### HNET Model UTIL ####################################

def build_model_hnet(hparams, num_input=2):
    if num_input == 2:
        # Build Dynamics Model and loss
        state_dim, out_dim = hparams.state_dim, hparams.out_dim

        # Specific environment model converts angle to [cos, sin]
        if hparams.env.startswith("inverted_pendulum") or hparams.env.startswith('cartpole') \
            or hparams.env == "door":
            state_dim = hparams.state_dim + 1
        elif hparams.env == "door_pose":
            state_dim = hparams.state_dim + 2
        input_dim = state_dim + hparams.control_dim
    else:
        input_dim, out_dim = hparams.in_dim, hparams.out_dim

    mnet = MLP(n_in=input_dim,
             n_out=out_dim, hidden_layers=hparams.h_dims, 
             no_weights=True, out_var=hparams.out_var,
             mlp_var_minmax=hparams.mlp_var_minmax)
    
    print('Constructed MLP with shapes: ', mnet.param_shapes)

    if hparams.model == "chunked_hnet":
        hnet = ChunkedHyperNetworkHandler(mnet.param_shapes,
                chunk_dim=hparams.chunk_dim,
                layers=hparams.hnet_arch,
                activation_fn=str_to_act(hparams.hnet_act),
                te_dim=hparams.emb_size,
                ce_dim=hparams.cemb_size,
        )
    else:
        #num_weights_class_net = MLP.shapes_to_num_weights(mnet.param_shapes)
        hnet = HyperNetwork(mnet.param_shapes,
                layers=hparams.hnet_arch,
                te_dim=hparams.emb_size,
                activation_fn=str_to_act(hparams.hnet_act)
        )
    init_params = list(hnet.parameters())

    # Calculate compression ratio
    if isinstance(hparams, Hparams):
        hparams.num_weights_class_hyper_net = sum(p.numel() for p in
                                        hnet.parameters() if p.requires_grad)
        hparams.num_weights_class_net = MainNetInterface.shapes_to_num_weights(mnet.param_shapes)
        hparams.compression_ratio_class = hparams.num_weights_class_hyper_net / \
                                                hparams.num_weights_class_net
    print('Created hypernetwork with ratio: ', hparams.compression_ratio_class)
    ### Initialize network weights.
    for W in init_params:
        if W.ndimension() == 1: # Bias vector.
            torch.nn.init.constant_(W, 0)
        elif hparams.hnet_init == "normal":
            torch.nn.init.normal_(W, mean=0, std=hparams.std_normal_init)
        elif hparams.hnet_init == "xavier":
            torch.nn.init.xavier_uniform_(W)

    if hasattr(hnet, 'chunk_embeddings'):
        for emb in hnet.chunk_embeddings:
            torch.nn.init.normal_(emb, mean=0, std=hparams.std_normal_cemb)

    if hparams.use_hyperfan_init:
        if hparams.model.startswith("hnet"):
            hnet.apply_hyperfan_init(temb_var=hparams.std_normal_temb**2)

    return mnet, hnet

def reload_model_hnet(hparams, task_id=None, num_input=2):
    mnet, hnet = build_model_hnet(hparams, num_input=num_input)
    # Restore Data
    collector = MonitorRL.resume_from_disk(hparams)
    agent = MPC(hparams, mnet, hnet=hnet, collector=collector)

    # Load Checkpoint
    if task_id is None:
        model_path = os.path.join(hparams.save_folder,
            f'TB{hparams.env}_{hparams.model}_{hparams.seed}', 'model.pt')
    else:
        model_path = os.path.join(hparams.save_folder,
            f'TB{hparams.env}_{hparams.model}_{hparams.seed}', f'model_{task_id}.pt')     

    checkpoint = torch.load(model_path, map_location=hparams.gpuid)
    print("Checkpoint Loaded")

    # Remove potentially unwanted data collector datas
    for tid in range(checkpoint['num_tasks_seen'], hparams.num_tasks):
        try:
            collector.states.pop(tid) 
            collector.actions.pop(tid)
            collector.nexts.pop(tid)
            collector.train_inds.pop(tid)
            collector.val_inds.pop(tid)

            collector.x_aggregate.pop(tid)
            collector.a_aggregate.pop(tid)
            collector.norms.pop(tid)
        except KeyError:
            pass

    # Add task embeddings to the hnet
    for task_id in range(checkpoint['num_tasks_seen']):
        hnet.add_task(task_id, hparams.std_normal_temb)
        if hparams.model == "hnet_si":
            si.si_register_buffer(mnet, hnet, task_id)
        if hparams.model == "hnet_ewc":
            ewc.ewc_register_buffer(mnet, hnet, task_id)

    mnet.load_state_dict(checkpoint['mnet_state_dict'])
    hnet.load_state_dict(checkpoint['hnet_state_dict'])
    # Load model to gpu
    mnet.to(hparams.gpuid)
    hnet.to(hparams.gpuid)
    print("Hnet restored")

    return mnet, hnet, agent, checkpoint, collector