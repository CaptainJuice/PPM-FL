import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : model save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_acc = -1

    def __call__(self, val_loss, acc, model, CV_idx,savedirname,test_acc):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_acc = test_acc
            self.save_checkpoint(val_loss, model, CV_idx,savedirname)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_acc= test_acc
            self.save_checkpoint(val_loss, model,CV_idx,savedirname)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, CV_idx,savedirname):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print('Acc On Test  = ',self.best_acc)
        # path = os.path.join(self.save_path, 'best_network.pth')
        # torch.save(model[0].state_dict(), path)	
        torch.save(model[0].state_dict(), f"{savedirname}/feature_net_CVidx={CV_idx}.pth")
        torch.save(model[1].state_dict(), f"{savedirname}/task_solver_CVidx={CV_idx}.pth")
        torch.save(model[2].state_dict(), f"{savedirname}/toporep_CVidx={CV_idx}.pth")
        self.val_loss_min = val_loss

