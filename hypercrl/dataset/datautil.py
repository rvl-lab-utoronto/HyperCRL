import torch
import numpy as np
import random
from torch.utils.data import TensorDataset, Dataset, random_split
from gym.envs.robotics.rotations import quat_mul, quat_conjugate

def rotate_imgs(data, r=None):
    if r is None:
        r = random.choice([0, 1, 2, 3])
    if (type(data) is tuple):
        img, label = data
        img = np.rot90(data[0], k=r, axes=(1, 2))
        return img, label, r
    if (type(data) is list):
        data[0] = np.rot90(data[0], k=r, axes=(1, 2))
        data += [r]
        return data
    else:
        img = np.rot90(data, k=r, axes=(1, 2))
        return (img, r)

def train_val_split(dataset, val_size=0.25):
    val_size = int(len(dataset) * 0.25)
    train_size = len(dataset) - val_size
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    return train_set, val_set

class DataCollector():
    def __init__(self, hparams):
        self.states = {}
        self.actions = {}
        self.nexts = {}
        self.train_inds = {}
        self.val_inds = {}

        self.x_aggregate = {}
        self.a_aggregate = {}
        self.dx_aggregate = {}
        self.norms = {}
        self.fig = None
        self.next_mode = hparams.dnn_out
        self.normalize_xu = hparams.normalize_xu
        self.env_name = hparams.env

    def num_tasks(self):
        return len(self.states)

    def update(self, existingAggregate, newValue):
        """
        compute new count, new mean, and new second momentum
        """
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2

        return (count, mean, M2)

    def preprocess(self, x_t, u, x_tt):
        """
        Process the pre-/post- processing of states
        """
        if self.env_name.startswith("inverted_pendulum") or self.env_name.startswith('cartpole'):
            if self.next_mode == "diff":
                x_tt = x_tt - x_t
            x_t = np.vstack(( x_t[0:1, :], np.cos(x_t[1:2, :]), np.sin(x_t[1:2, :]), x_t[2:, :] ))
        elif self.env_name in ["half_cheetah_body", "hopper"]:
            if self.next_mode == "diff":
                x_tt = np.vstack((x_tt[0:1, :], x_tt[1:, :] - x_t[1:, :]))
            x_t = np.vstack(( x_t[1:2, :], np.cos(x_t[2:3, :]), np.sin(x_t[2:3, :]), x_t[3:, :] ))
        elif self.env_name == "door":
            if self.next_mode == "diff":
                x_tt = x_tt - x_t
            x_t = np.vstack(( x_t[0:-1, :], np.cos(x_t[-1:, :]), np.sin(x_t[-1:, :]) ))
        elif self.env_name == "door_pose":
            if self.next_mode == "diff":
                quat_diff = quat_mul(x_tt[3:7, :].T, quat_conjugate(x_t[3:7, :].T)).T
                x_tt = x_tt - x_t
                x_tt[3:7, :] = quat_diff

            x_t = np.vstack(( x_t[0:-2, :], np.cos(x_t[-2:-1, :]), np.sin(x_t[-2:-1, :]),
                    np.cos(x_t[-1:, :]), np.sin(x_t[-1:, :])))
        else:
            if self.next_mode == "diff":
                x_tt = x_tt - x_t

        return x_t, u, x_tt

    def add(self, x_t, u, x_tt, task_id):
        # Convert Format
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        if x_t.ndim == 1:
            x_t = x_t[:, None]
        if x_tt.ndim == 1:
            x_tt = x_tt[:, None]
        if u.ndim == 1:
            u = u[:, None]

        x_t, u, x_tt = self.preprocess(x_t, u, x_tt)

        if task_id in self.states:
            self.states[task_id].append(x_t)
            self.actions[task_id].append(u)
            self.nexts[task_id].append(x_tt)
            if self.normalize_xu:
                self.x_aggregate[task_id] = self.update(self.x_aggregate[task_id], x_t)
                self.a_aggregate[task_id] = self.update(self.a_aggregate[task_id], u)
        else:
            self.states[task_id] = [x_t]
            self.actions[task_id] = [u]
            self.nexts[task_id] = [x_tt]
            if self.normalize_xu:
                self.x_aggregate[task_id] = self.update((0, 0, 0), x_t)
                self.a_aggregate[task_id] = self.update((0, 0, 0), u)
        # Train or val
        is_train = (random.random() <= 0.75)
        ind = len(self.states[task_id]) - 1
        if is_train:
            if task_id in self.train_inds:
                self.train_inds[task_id].append(ind)
            else:
                self.train_inds[task_id] = [ind]
        else:
            if task_id in self.val_inds:
                self.val_inds[task_id].append(ind)
            else:
                self.val_inds[task_id] = [ind]

    def finalize(self, task_id):
        def one(existingAggregate):
            (count, mean, M2) = existingAggregate
            if count < 2:
                return float('nan')
            else:
                sample_var = M2 / (count - 1)
                std = np.sqrt(sample_var)
                mean, std = torch.FloatTensor(mean).T, torch.FloatTensor(std).T
                std[std < 1e-9] = 1
                return mean, std

        x_mu, x_std = one(self.x_aggregate[task_id])
        a_mu, a_std = one(self.a_aggregate[task_id])

        self.norms[task_id] = (x_mu, x_std, a_mu, a_std)

        return self.norms[task_id]
    
    def norm(self, task_id):
        """
        Return x_mu, x_std, a_mu, a_std, (dx_mu, dx_std) of the given task_id
        """
        return self.norms[task_id]

    def get_dataset(self, task_id, ds_range=None):
        """
        Return a pytorch dataset of (state, actions, next_state)
        states, actions are normalized to N(0, 1)
        """

        states = torch.FloatTensor(np.hstack(self.states[task_id])).T
        actions = torch.FloatTensor(np.hstack(self.actions[task_id])).T
        nexts = torch.FloatTensor(np.hstack(self.nexts[task_id])).T

        # Get Norm and normalize
        if self.normalize_xu:
            x_mu, x_std, a_mu, a_std = self.finalize(task_id)
            states = (states - x_mu) / x_std
            actions = (actions - a_mu) / a_std
            if self.next_mode != "diff":
                nexts = (nexts - x_mu) / x_std

        train_inds = self.train_inds[task_id]
        val_inds = self.val_inds[task_id]

        if ds_range == "second_half":
            train_inds = train_inds[len(train_inds) // 2:]
        train_set = TensorDataset(states[train_inds], actions[train_inds], nexts[train_inds])
        val_set = TensorDataset(states[val_inds], actions[val_inds], nexts[val_inds])

        return train_set, val_set

    def sizes(self):
        N = []
        for x in self.states:
            N.append(len(self.states[x]))
        return N

class Split_Class_Dataset(Dataset):
    """
    New Disjoint Class Setting

    Given a dataset (train or val or test),
    Return a dataset that can set new tasks
    """
    def __init__(self, data, num_task = 1, rot=False):
        self.data = data
        self.rot = rot
        self.map = {}
        self.task_id = 0
        self.num_task = num_task
        self.compile()
    
    def compile(self):
        for i, (X, y) in enumerate(self.data):
            if y not in self.map:
                self.map[y] = [i]
            else:
                self.map[y].append(i)

        # Divide the all classes in the dataset 
        # to a number of tasks
        total = len(self.map.keys())
        per_task = int(total // self.num_task)
        index = [i for i in range(total)]
        # TODO: randomize index selection per group
        
        # Map each task to a set of data indices
        # by joining the tasks
        task_inds = {}
        for id in range(self.num_task):
            base = id * per_task
            task_inds[id] = []
            for i in range(base, base+per_task):
                task_inds[id] += self.map[index[i]]
        
        task_inds[self.num_task] = np.arange(len(self.data))
        
        self.map = task_inds

    def set_task(self, id):
        if id == "all":
            id = self.num_task
        if id not in self.map:
            raise ValueError("Task ID not found")
        
        self.task_id = id

    def __getitem__(self, index):
        ind = self.map[self.task_id][index]
        data = self.data[ind]
        if self.rot:
            data = rotate_imgs(data)
        return data

    def __len__(self):
        return len(self.map[self.task_id])

class Mixed_Dataset(Dataset):

    def __init__(self, datas, mode):
        assert hasattr(datas, '__len__')
        assert hasattr(datas[0], '__len__')
        
        self.mode = mode
        self.datas = datas
        self.lens = [len(data) for data in self.datas]
        self.total = sum(self.lens)

        self.map = []
        ds_ind = 0
        ds_sum = 0
        for i in range(self.total):
            ind = i - ds_sum
            while ind >= self.lens[ds_ind]:
                ds_sum += self.lens[ds_ind]
                ds_ind += 1
                ind = i - ds_sum
            self.map.append((ds_ind, ind))

        if mode == "iid":
            np.random.shuffle(self.map)      

    def __getitem__(self, index):
        if self.mode == "iid":
            ds, ind = self.map[index]
            return self.datas[ds][ind]
        elif self.mode == "sequential":
            ds, ind = self.map[index]
            return self.datas[ds][ind]

    def __len__(self):
        return self.total        