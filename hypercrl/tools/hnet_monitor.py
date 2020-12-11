import torch
from .tools import MonitorRL

class MonitorHnet(MonitorRL):
    def __init__(self, hparams, agent, mnet, hnet, collector):
        super(MonitorHnet, self).__init__(hparams, agent, mnet, collector, None)
        self.mnet = mnet
        self.hnet = hnet
        self.model_to_save = {'mnet': mnet, 'hnet': hnet}

        self.loss_task = 0
        self.loss_reg = 0

    def train_step(self, loss_task, loss_reg, dTheta, grad_tloss, weights):
        self.loss_task += loss_task.item()
        self.loss_reg += loss_reg.item()
        if (self.train_iter % self.print_train_every == 0):
            self.loss_task /= self.print_train_every
            self.loss_reg /= self.print_train_every
            loss_tot = self.loss_reg + self.loss_task
            print(f"Batch: {self.train_iter}, Loss: {loss_tot:.5f}, " + 
                  f"Task L: {self.loss_task:.5f}, Reg L: {self.loss_reg:.5f}")

            i = self.train_iter

            self.writer.add_scalar('train/mse_loss', self.loss_task, i)
            self.writer.add_scalar('train/regularizer', self.loss_reg, i)
            self.writer.add_scalar('train/full_loss', loss_tot, i)
            if dTheta is not None:
                dT_norm = torch.norm(torch.cat([d.view(-1) for d in dTheta]), 2)
                self.writer.add_scalar('train/dTheta_norm', dT_norm, i)
            if grad_tloss is not None:
                (grad_tloss, grad_full, grad_diff_norm, grad_cos) = grad_tloss
                self.writer.add_scalar('train/full_grad_norm',
                                  torch.norm(grad_full, 2), i)
                self.writer.add_scalar('train/reg_grad_norm',
                                  grad_diff_norm, i)
                self.writer.add_scalar('train/cosine_task_reg',
                                  grad_cos, i)
            
            self.loss_task = 0
            self.loss_reg = 0

        if (self.train_iter % self.log_hist_every == 0):
            for i, weight in enumerate(weights):
                self.writer.add_histogram(f'train/weight/{i}', weight.flatten(), self.train_iter)
        self.train_iter += 1

    def data_aggregate_step(self, x_tt, task_id, it):
        if self.hparams.env == "lqr10":
            l2_pos = np.linalg.norm(x_tt[:10])
            l2_vel = np.linalg.norm(x_tt[10:])
            self.writer.add_scalar(f'lqr10/{task_id}/l2_pos', l2_pos, it)
            self.writer.add_scalar(f'lqr10/{task_id}/l2_vel', l2_vel, it)

    def validate_task(self, task_id, loader, mll, is_training=False):
        self.mnet.eval()
        self.hnet.eval()
        gpuid = self.hparams.gpuid
        
        # Initialize Stats
        val_loss = 0
        val_diff = 0
        N = len(loader)
        
        with torch.no_grad():
            weights = self.hnet.forward(task_id)

            for _, data in enumerate(loader):
                x_t, a_t, x_tt = data
                x_t, a_t, x_tt = x_t.to(gpuid), a_t.to(gpuid), x_tt.to(gpuid)
                X = torch.cat((x_t, a_t), dim=-1)
                
                Y = self.mnet.forward(X, weights)
                
                loss = mll(Y, x_tt, weights)
                if self.hparams.out_var:
                    Y, _ = torch.split(Y, Y.size(-1)//2, dim=-1)
                diff = torch.abs(Y - x_tt).mean(dim=0)
                
                val_loss += loss
                val_diff += diff
            
            val_loss = val_loss / N
            val_diff = val_diff / N

        print(f"Iter {self.train_iter}, Task: {task_id}, " + \
              f"Val Loss: {val_loss.item():.5f}, Val Diff: {val_diff.mean().item()}")
        
        return val_loss, val_diff