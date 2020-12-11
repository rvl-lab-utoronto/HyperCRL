import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset

def dataset2tensor(dataset):
    return TensorDataset(*dataset[:])

class PNNBlock(nn.Module):
    def __init__(self, n_in, n_out, i, k):
        super(PNNBlock, self).__init__()
        self.W = nn.Linear(n_in, n_out)
        self.k = k
        self.i = i
        if i > 0 and k > 0:
            self.V = nn.Linear(n_in * k, n_out)
            self.alpha = nn.Parameter(torch.randn(n_in * k) * 0.01)
            self.U = nn.Linear(n_out, n_out)

    def forward(self, h):
        if self.i == 0:
            # First Hidden Layer
            x = self.W(h[0])
        else:
            # Subsequent layer
            x = self.W(h[self.k])

        if self.i > 0 and self.k > 0:
            a = torch.cat(h[:self.k], dim=1)
            a = self.V(self.alpha * a)
            a = F.relu(a)
            a = self.U(a)
            x = x + a

        return F.relu(x)

class PNN(nn.Module):
    def __init__(self, hparams):
        super(PNN, self).__init__()
        self.x_dim = hparams.state_dim
        self.a_dim = hparams.control_dim

        # Dimensions of the last weight layer
        self.out_dim = hparams.out_dim
        self.out_var = hparams.out_var

        self.num_tasks = 0

        self.h_dims = hparams.h_dims
        self.columns = nn.ModuleList([])

    def freeze(self, k):
        for p in self.columns[k].parameters():
            p.requires_grad = False

    def add_weights(self, task_id):
        if task_id < self.num_tasks:
            return
        n_in = self.x_dim + self.a_dim
        col = nn.ModuleList()
        for i, n_out in enumerate(self.h_dims):
            h = PNNBlock(n_in, n_out, i, task_id)
            n_in = n_out
            col.append(h)

        # Output Block
        col.append(nn.Linear(n_in, self.out_dim))
        if self.out_var:
            linear_var = nn.Linear(n_in, self.out_dim)
            col.append(linear_var)
            max_logvar = nn.Parameter(torch.ones(1, self.out_dim) / 2.0)
            min_logvar = nn.Parameter(-torch.ones(1, self.out_dim) * 10.0)
            self.register_parameter(f'max_logvar_{task_id}', max_logvar)
            self.register_parameter(f'min_logvar_{task_id}', min_logvar)

        # Append the column
        self.columns.append(col)

        self.num_tasks += 1

    def forward(self, x, a, task_id = None):

        if task_id is None:
            task_id = self.num_tasks - 1

        h = [torch.cat((x, a), dim=-1)]

        for i in range(len(self.h_dims)):
            o = []
            for j in range(task_id + 1):
                lat = self.columns[j][i]
                o.append(lat(h))
            h = o
        h = h[task_id]

        if self.out_var:
            linear1 = self.columns[task_id][-2]
            linear2 =  self.columns[task_id][-1]
            max_logvar = getattr(self, f'max_logvar_{task_id}')
            min_logvar = getattr(self, f'min_logvar_{task_id}')
            output = linear1(h)
            logvar = linear2(h)
            logvar = max_logvar - F.softplus(max_logvar - logvar)
            logvar = min_logvar + F.softplus(logvar - min_logvar)
            output = torch.cat((output, logvar), dim=-1)
            return output
        else:
            out_block = self.columns[task_id][-1]
            return out_block(h)

    def replay(self, i):
        return torch.zeros(1), torch.zeros(1)

class Baseline(nn.Module):
    def __init__(self, hparams):
        super(Baseline, self).__init__()

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

    def reinit(self):
        def reset(m):
            if type(m) == nn.Linear:
                m.reset_parameters()
        self.net.apply(reset)

    def add_inducing_points(self, z, task_id):
        if hasattr(self, f"z_x{task_id}"):
            z_x, z_a, z_y = self.get_inducing_points(task_id)
            z_x = torch.cat((z_x, z[0].to(device=z_x.device, dtype=z_x.dtype)), dim=0)
            z_a = torch.cat((z_a, z[1].to(device=z_a.device, dtype=z_a.dtype)), dim=0)
            z_y = torch.cat((z_y, z[2].to(device=z_y.device, dtype=z_y.dtype)), dim=0)
        else:
            z_x, z_a, z_y = z[0], z[1], z[2]
        
        self.register_buffer(f"z_x{task_id}", z_x)
        self.register_buffer(f"z_a{task_id}", z_a)
        self.register_buffer(f"z_y{task_id}", z_y)

    def get_inducing_points(self, task_id):
        z_x = getattr(self, f"z_x{task_id}")
        z_a = getattr(self, f"z_a{task_id}")
        z_y = getattr(self, f"z_y{task_id}")
        return z_x, z_a, z_y

    def replay(self, task_id):
        x, a, y = self.get_inducing_points(task_id) 

        pred = self.forward(x, a, task_id)
        return pred, y

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

    def freeze(self, task_id):
        if self.out_var:
            max_logvar = getattr(self, f'max_logvar_{task_id}')
            min_logvar = getattr(self, f'min_logvar_{task_id}')
            max_logvar.requires_grad = False
            min_logvar.requires_grad = False

class MTBaseline(Baseline):
    def __init__(self, hparams):
        super(MTBaseline, self).__init__(hparams)

        self.gpuid = hparams.gpuid
        self.bs = hparams.bs
        self.old_data, self.old_data_iter = [], []
        self.counter = 0

    def add_trainset(self, collector, task_id):
        old_data, old_data_iter = [], []
        for tid in range(0, 1+ task_id):
            train_set, _ = collector.get_dataset(tid)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.bs,
                shuffle=True, drop_last=True)
            old_data.append(train_loader)
            old_data_iter.append(iter(train_loader))

        self.old_data = old_data
        self.old_data_iter = old_data_iter

    def replay(self, task_id):
        loader_it = self.old_data_iter[task_id]
        try:
            data = next(loader_it)
        except StopIteration:
            # Reset the dataloader iterable
            loader_it = iter(self.old_data[task_id])
            self.old_data_iter[task_id] = loader_it
            data = next(loader_it)

        x_t, a_t, x_tt = data
        x_t, a_t, x_tt = x_t.to(self.gpuid), a_t.to(self.gpuid), x_tt.to(self.gpuid)

        pred = self.forward(x_t, a_t, task_id)
        return pred, x_tt

class MSELoss(nn.Module):
    def __init__(self, model, M=0, task_id = 0, reg_lambda=0, out_var=False):
        super(MSELoss, self).__init__()
        self.task_id = task_id
        self.model = model
        self.reg_norm = 1
        self.reg_lambda = reg_lambda
        self.M = M
        self.out_var = out_var

    def regularize(self, task_id=None):
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

    def forward(self, pred, y, task_id = None):
        mse = nn.functional.mse_loss
        y_dim = y.size(-1)

        # Compute loss of a particular task (typically during evaluation)
        if task_id is not None or self.M == 0:
            loss = mse(pred, y, reduction='sum') / y_dim

        # Compute loss across all task
        else:
            loss = mse(pred, y, reduction='sum') / y_dim

            for i in range(self.task_id):
                # Adjust the loss according to batch size
                a = mse(*self.model.replay(i), reduction='sum') / y_dim
                a = (a  * y.size(0) / self.M).to(loss.device)
                loss = loss + a

        reg = self.regularize(task_id)

        return - loss - reg

class LogProbLoss(MSELoss):
    def __init__(self, model, M=0, task_id = 0, reg_lambda=0):
        super(LogProbLoss, self).__init__(model, M, task_id, reg_lambda)

    def forward(self, pred, y, task_id = None):
        y_dim = y.size(-1)

        if task_id is None:
            tid = self.model.num_tasks - 1
        else:
            tid = task_id

        def one_task(tid, pred, y):
            mu, logvar = torch.split(pred, y_dim, dim=-1)

            max_logvar = getattr(self.model, f'max_logvar_{tid}')
            min_logvar = getattr(self.model, f'min_logvar_{tid}')

            # Compute loss of a task (i.e during evaluation)
            inv_var = torch.exp(-logvar)
            loss = ((mu - y) ** 2) * inv_var + logvar
            loss = loss.sum() / y_dim

            # Regularize max/min var
            loss += 0.01 * (max_logvar.sum() - min_logvar.sum())

            return loss
        
        loss = one_task(tid, pred, y)
        if task_id is None and self.M > 0:
            for i in range(self.task_id):
                # Adjust the loss according to batch size
                pred_coreset, y_coreset = self.model.replay(i)
                a = one_task(tid, pred_coreset, y_coreset)
                a = (a  * y.size(0) / self.M).to(loss.device)
                loss = loss + a

        reg = self.regularize(task_id)

        return - loss - reg

class IPSelector():
    def __init__(self, dataset, hparams):
        self.dataset = dataset2tensor(dataset)
        self.M = hparams.M
        self.N = len(self.dataset)
        if self.M == -1:
            self.M = self.N

        # Index of the inducing points in dataset (initialized randomly)
        self.points_ind = self.rand_indices()

    def rand_indices(self):
        indices = random.sample(range(self.N), self.M)
        return indices

    def inducing_points(self):
        z = self.dataset[self.points_ind]
        return z


class BoundaryTest():
    def __init__(self, model, hparams):
        self.model = model
        self.hparams = hparams
        self.kls = []
        self.t_stats = []
        self.pvalues = []
        self.T = 20
        self.alpha = 0.005

        # Plot
        # plt.ion()

        #f, (y1_ax, y2_ax) = plt.subplots(2, 1, figsize=(6, 4))
        #self.fig = (f, (y1_ax, y2_ax))

        #f.show()
        #f.canvas.draw()

    def factor_batch(self, p):
        N = p.mean.size(0)
        K = p.mean.size(1)

        dists = []

        for n in range(N):
            mu = p.mean[n, :]

            # Get the factorized covariance matrix for each data point in minibatch
            covar = torch.eye(K, dtype=mu.dtype, device=mu.device)
            ind = torch.arange(n, K*N, N)
            covar *= p.covariance_matrix[ind, ind]
            mvn = torch.distributions.multivariate_normal. \
                    MultivariateNormal(mu, covar)
            dists.append(mvn)
        return dists

    def sym_kl(self, x, a):
        """
        Compute the symmetric KL divergence between GP prior and posterior
        """
        with torch.no_grad():
            q_f = self.model(x, a, prior=False)
            p_f = self.model(x, a, prior=True)

            q_f = self.factor_batch(q_f)
            p_f = self.factor_batch(p_f)

            skl = []
            for (p, q) in zip(q_f, p_f):
                skl_i = 0.5 * torch.distributions.kl.kl_divergence(p, q) \
                     + 0.5 * torch.distributions.kl.kl_divergence(q, p)
                skl.append(skl_i)
            
        return torch.tensor(skl)

    def test(self, x, a):
        if self.hparams.model != "GP":
            return False
        
        kl = self.sym_kl(x, a)
        self.kls.append(kl)

        if len(self.kls) == 1:
            return False

        a = self.kls[-2].cpu().numpy()
        b = self.kls[-1].cpu().numpy()
        t_stat, pvalue = scipy.stats.ttest_ind(a, b, axis=0, equal_var=False)
        self.t_stats.append(t_stat)
        self.pvalues.append(pvalue)

        reject = (pvalue <= self.alpha)
        return reject

    def plot(self):
        f, (y1_ax, y2_ax) = self.fig

        ts = list(range(1, len(self.kls)))

        kl = [self.kls[i].mean().item() for i in ts]
        y1_ax.plot(ts, kl, 'k')
        y2_ax.plot(ts, self.t_stats, 'k')

        y1_ax.set_title("KL")
        y2_ax.set_title("Boundary Test Statistic")
        f.canvas.draw()
