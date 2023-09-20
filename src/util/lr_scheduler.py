import numpy as np
import torch.optim.lr_scheduler

class CustomLRSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, start_lr, max_lr, ramp_up_epochs, last_epoch=-1):
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.ramp_up_epochs = ramp_up_epochs
        super(CustomLRSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.ramp_up_epochs:
            lr = self.start_lr + (self.max_lr - self.start_lr) * (self.last_epoch / self.ramp_up_epochs)
        else:
            decay_factor = np.exp(-0.1 * (self.last_epoch - self.ramp_up_epochs))
            lr = self.max_lr * decay_factor
        return [lr for _ in self.base_lrs]