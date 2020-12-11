#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import functional as F

from ..mnet_interface import MainNetInterface
from ..hyper_model import HyperNetwork

#------------- "Synaptic Intelligence Synapses"-specifc functions -------------#

def si_register_buffer(mnet, hnet, task_id):
    """
    Register buffers for si and inital weights
    This method should be called at the start of each task
    or when before reloading the model
    """
    # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
    mnet.si_w = []
    mnet.si_p_old = []
    mnet.si_grad = []

    weights = hnet.forward(task_id)

    mnet.si_p_start = []

    # Register si omega buffer in mnet
    for j, shape in enumerate(mnet.param_shapes):
        p = weights[j]
        buf = torch.zeros(shape)
        mnet.register_buffer(f'{j}_SI_omega_t_{task_id}', p.detach().clone().zero_())

        mnet.si_w.append( p.detach().clone().zero_())
        mnet.si_p_old.append(p.detach().clone())
        mnet.si_p_start.append(p.detach().clone())
        mnet.si_grad.append(p.detach().clone().zero_())

def si_update_optim_step(mnet, weights, task_id):
    for j in range(len(mnet.si_w)):
        p = weights[j]
        grad = mnet.si_grad[j]
        if p.requires_grad:
            mnet.si_w[j].add_(-grad*(p.detach()-mnet.si_p_old[j]))
            mnet.si_p_old[j] = p.detach().clone()

def si_update_grad(mnet, weights, task_id):
    for j in range(len(mnet.si_w)):
        p = weights[j]
        if p.requires_grad and p.grad is not None:
            mnet.si_grad[j] = p.grad

def update_omega(mnet, hnet, epsilon, task_id):
    '''After completing training on a task, update the per-parameter regularization strength.
    [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

    # Get current weights after training
    weights = hnet.forward(task_id)

    # get si_w
    si_w = mnet.si_w

    for j in range(len(si_w)):
        p_change = weights[j].detach() - mnet.si_p_start[j]
        omega = torch.abs(si_w[j])/(p_change**2 + epsilon)
        mnet.register_buffer(f'{j}_SI_omega_t_{task_id}', omega)

def get_si_omega(mnet, current_task_id):
    omegas = []
    for task_id in range(current_task_id):
        omega = []
        for j in range(len(mnet.param_shapes)):
            omega.append(getattr(mnet, f'{j}_SI_omega_t_{task_id}'))
        omegas.append(omega)

    return omegas