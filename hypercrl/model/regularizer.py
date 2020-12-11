"""
Code adopted from https://github.com/GMvandeVen/continual-learning
Paper: https://arxiv.org/abs/1904.07734
"""

import abc
import numpy as np
import torch
import copy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from .mbrl import LogProbLoss

#############################
## Data-handling functions ##
#############################

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
}

def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False, augment=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_, batch_size=batch_size, shuffle=True,
        collate_fn=(collate_fn or default_collate), drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier.
    Adds methods for "context-dependent gating" (XdG), "elastic weight consolidation" (EWC) and
    "synaptic intelligence" (SI) to its subclasses.'''

    def __init__(self):
        super().__init__()
 
    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass


    #----------------- XdG-specifc functions -----------------#

    def apply_XdGmask(self, task):
        '''Apply task-specific mask, by setting activity of pre-selected subset of nodes to zero.
        [task]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        # Loop over all buffers for which a task-specific mask has been specified
        for i,excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[task][i]] = 0.      # -> find task-specifc mask
            excit_buffer.set_(torchType.new(gating_mask))  # -> apply this mask

    def reset_XdGmask(self):
        '''Remove task-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1., len(excit_buffer))  # -> define "unit mask" (i.e., no masking at all)
            excit_buffer.set_(torchType.new(gating_mask))   # -> apply this unit mask


    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, dataset, allowed_classes=None, collate_fn=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index, data in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            if len(data) == 3:
                x, a, y = data
                x = x.to(self._device())
                a = a.to(self._device())
                y = y.to(self._device())
                output = self(x, a)
            else:
                x, y = data
                x = x.to(self._device())
                y = y.to(self._device())
                output = self(x)
            if self.empircal_fisher:
                if self.out_var:
                    mu, logvar = torch.split(output, output.size(-1)//2, dim=-1)
                else:
                    mu = output
                    logvar = torch.zeros_like(mu)
                inv_var = torch.exp(-logvar)
                nll = 0.5 * ((mu - y) ** 2) * inv_var + 0.5 * logvar
                negloglikelihood = nll.sum()
            else:
                # In this case, compute the true fisher
                # F_ii = E_x {\sum_j [(grad_theta_i u_j)^2/sigma_j]}
                negloglikelihood = output.sum() # Assume unit covariance

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])
        #print([n if p.requires_grad else 0 for n, p in self.named_parameters()])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)


    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count+1):
                for n, p in self.named_parameters():
                    if n.startswith('max_logvar') or n.startswith('min_logvar') or n.startswith('w_'):
                        continue
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())


    def ewc_register_buffer(self, task_count):
        """ Register empty ewc buffer for reloading model"""
        self.EWC_task_count = task_count
        for task in range(1, self.EWC_task_count+1):
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    # -mode (=MAP parameter estimate)
                    self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else task),
                                        torch.zeros_like(p))
                    # -precision (approximated by diagonal Fisher Information matrix)
                    self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task),
                                        torch.zeros_like(p))
    
    #------------- "Synaptic Intelligence Synapses"-specifc functions -------------#

    def si_register_buffer(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
                self.register_buffer('{}_SI_omega'.format(n), p.data.clone().zero_())

    def si_zero_stats(self):
        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        self.W = {}
        self.p_old = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.W[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()

    def si_update(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    self.W[n].add_(-p.grad*(p.detach()-self.p_old[n]))
                self.p_old[n] = p.detach().clone()

    def update_omega(self, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        W = self.W
        for n, p in self.named_parameters():
            if n.startswith('max_logvar') or n.startswith('min_logvar') or n.startswith('w_'):
                continue
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)


    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if n.startswith('max_logvar') or n.startswith('min_logvar') or n.startswith('w_'):
                    continue
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

class BaselineReg(ContinualLearner):
    def __init__(self, hparams):
        super(BaselineReg, self).__init__()

        x_dim = hparams.state_dim
        a_dim = hparams.control_dim

        # Hiden Layers
        layers = []
        in_dim = x_dim + a_dim
        for i, dim in enumerate(hparams.h_dims):
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = dim
        self.net = nn.Sequential(*layers)

        # Dimensions of the last weight layer
        self.h_dim = in_dim
        self.out_dim = hparams.out_dim
        self.out_var = hparams.out_var
        self.num_tasks = 0
        self.reg_name = hparams.model

        # -EWC:
        if hparams.model == "ewc":
            self.empircal_fisher = hparams.empircal_fisher
            self.gamma = hparams.ewc_online_gamma  #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
            self.online = hparams.ewc_online      #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
            self.fisher_n = None    #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
            self.EWC_task_count = 0 #-> keeps track of number of quadratic loss terms (for "offline EWC")

    def add_weights(self, task_id):
        if task_id < self.num_tasks:
            return

        linear = nn.Linear(self.h_dim, self.out_dim)
        self.add_module(f'w_{task_id}', linear)

        if self.out_var:
            linear_var = nn.Linear(self.h_dim, self.out_dim)
            self.add_module(f'w_{task_id}_2', linear_var)
            max_logvar = nn.Parameter(torch.ones(1, self.out_dim) / 2.0)
            min_logvar = nn.Parameter(-torch.ones(1, self.out_dim) * 10.0)

            self.register_parameter(f'max_logvar_{task_id}', max_logvar)
            self.register_parameter(f'min_logvar_{task_id}', min_logvar)

        self.num_tasks += 1

    def forward(self, x, a, task_id=None):
        if task_id is None:
            task_id = self.num_tasks - 1

        phi = self.net(torch.cat((x, a), dim=-1))
        linear = getattr(self, f'w_{task_id}')
        output = linear(phi)

        if self.out_var:
            linear2 = getattr(self, f'w_{task_id}_2')
            max_logvar = getattr(self, f'max_logvar_{task_id}')
            min_logvar = getattr(self, f'min_logvar_{task_id}')
            logvar = linear2(phi)
            logvar = max_logvar - F.softplus(max_logvar - logvar)
            logvar = min_logvar + F.softplus(logvar - min_logvar)
            output = torch.cat((output, logvar), dim=-1)

        return output

class BaselineCls(ContinualLearner):
    def __init__(self, hparams):
        super(BaselineCls, self).__init__()

        in_dim = hparams.in_dim
        self.in_dim = in_dim

        # Hiden Layers
        layers = []
        for i, dim in enumerate(hparams.h_dims):
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = dim
        self.net = nn.Sequential(*layers)

        # Dimensions of the last weight layer
        self.h_dim = in_dim
        self.out_dim = hparams.out_dim
        self.num_tasks = 0
        self.reg_name = hparams.model

        # -EWC:
        if hparams.model == "ewc":
            self.empircal_fisher = hparams.empircal_fisher
            self.gamma = hparams.ewc_online_gamma  #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
            self.online = hparams.ewc_online      #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
            self.fisher_n = None    #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
            self.EWC_task_count = 0 #-> keeps track of number of quadratic loss terms (for "offline EWC")

    def add_weights(self, task_id):
        if task_id < self.num_tasks:
            return

        linear = nn.Linear(self.h_dim, self.out_dim)
        self.add_module(f'w_{task_id}', linear)
        self.num_tasks += 1

    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = self.num_tasks - 1

        phi = self.net(x.view(-1, self.in_dim))
        linear = getattr(self, f'w_{task_id}')
        output = linear(phi)

        return F.log_softmax(output)

    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, dataset, allowed_classes=None, collate_fn=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            if self.empircal_fisher:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y)==int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])
        #print([n if p.requires_grad else 0 for n, p in self.named_parameters()])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)

class CLRegLoss(nn.Module):
    def __init__(self, model, task_id = 0, reg_lambda=0, out_var = False):
        super(CLRegLoss, self).__init__()
        self.task_id = task_id
        self.model = model
        self.out_var = out_var
        self.reg_norm = 1
        self.reg_lambda = reg_lambda

    def regularize(self):
        if self.reg_lambda == 0:
            return 0

        l_reg = None
        mp = self.model.named_parameters()
        
        for name, W in mp:
            if name.startswith('max_logvar') or name.startswith('min_logvar'):
                continue
            if l_reg is None:
                l_reg = W.norm(self.reg_norm)
            else:
                l_reg = l_reg + W.norm(self.reg_norm)

        l_reg = self.reg_lambda * l_reg
        return l_reg

    def prob_loss(self, pred, y, task_id=None):
        y_dim = y.size(-1)

        mu, logvar = torch.split(pred, y_dim, dim=-1)

        max_logvar = getattr(self.model, f'max_logvar_{self.task_id}')
        min_logvar = getattr(self.model, f'min_logvar_{self.task_id}')

        # Compute loss of a task (i.e during evaluation)
        inv_var = torch.exp(-logvar)
        loss = ((mu - y) ** 2) * inv_var + logvar
        loss = loss.sum() / y_dim

        # Regularize max/min var
        loss += 0.01 * (max_logvar.sum() - min_logvar.sum())

        return loss

    def mse_loss(self, pred, y, task_id=None):
        y_dim = y.size(-1)

        loss = (pred-y)**2
        loss = loss.sum() / y_dim

        return loss

    def task_loss(self, pred, y, task_id=None):
        if self.out_var:
            loss = self.prob_loss(pred, y, task_id)
        else:
            loss = self.mse_loss(pred, y, task_id)

        return loss

class EWCLoss(CLRegLoss):
    def __init__(self, model, task_id = 0, reg_lambda=0, ewc_beta=0, out_var=False):
        super(EWCLoss, self).__init__(model, task_id, reg_lambda, out_var)
        self.ewc_beta = ewc_beta

    def forward(self, pred, y, task_id=None):
        loss = self.task_loss(pred, y, task_id)

        # EWC
        if self.task_id > 0:
            ewc_loss = self.ewc_beta * self.model.ewc_loss()
            loss += ewc_loss

        reg = self.regularize()

        return -loss - reg

class SILoss(CLRegLoss):
    def __init__(self, model, task_id = 0, reg_lambda=0, si_c=0, out_var=False):
        super(SILoss, self).__init__(model, task_id, reg_lambda, out_var)
        self.si_c = si_c

    def forward(self, pred, y, task_id=None):
        loss = self.task_loss(pred, y, task_id)

        # SI
        if self.task_id > 0:
            si_loss = self.si_c * self.model.surrogate_loss()
            loss += si_loss

        reg = self.regularize()

        return -loss - reg
