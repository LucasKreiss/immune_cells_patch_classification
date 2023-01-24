import numpy as np
import torch
import matplotlib.pyplot as plt
from plot_training_curves import plot_example_image


class Trainer_classify:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 criterion_pos: bool = False,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 n_steps_progressbar = 10
                 ):

        self.model = model
        self.criterion = criterion
        self.criterion_pos = criterion_pos
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.n_steps_progressbar = n_steps_progressbar

        self.training_loss_mean = []
        self.training_loss_std = []
        self.training_loss_min = []
        self.training_loss_max = []
        self.validation_loss_mean = []
        self.validation_loss_std = []
        self.validation_loss_min = []
        self.validation_loss_max = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = range(self.epochs)#trange(self.epochs, desc='Progress')
        
        for i in progressbar:
            
            if i>0:
                if (i%round(self.epochs/self.n_steps_progressbar) == 0):
                    print('Epochs: ',self.epoch,'/',self.epochs)
            else:
                print('Epochs: ',0,'/',self.epochs)
                
            
            
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss_mean, self.training_loss_std, self.training_loss_min, self.training_loss_max, self.validation_loss_mean, self.validation_loss_std, self.validation_loss_min, self.validation_loss_max, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        #batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
        #                  leave=False)
        batch_iter = enumerate(self.training_DataLoader)

        
        for i, (x, y) in batch_iter:
            
            
            if isinstance(x,torch.Tensor):
                inp  = x.to(self.device)  # send to device (GPU or CPU)
                inp = torch.squeeze(inp)
                
            if isinstance(y,torch.Tensor):
                target = y.to(self.device)  # send to device (GPU or CPU)
                target = torch.squeeze(target)
            else:
                target=y
                
                
            
            
            self.optimizer.zero_grad()  # zerograd the parameters
            
            out = torch.squeeze(self.model(inp))  # one forward pass         
            
            
            
            if (self.criterion_pos==False):
                loss = self.criterion(out, target) # calculate loss
            else:
                loss = 1 - self.criterion(out, target)  # calculate loss
                   
                
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            #batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  
            # update progressbar every n steps
            #if i>0:
            #    if (i%round(len(self.training_DataLoader)/self.n_steps_progressbar) == 0):
            #        print('\tTrain:',i,'/',len(self.training_DataLoader))
            #else:
            #    print('\tTrain:',0,'/',len(self.training_DataLoader))
            
        self.training_loss_mean.append(np.mean(train_losses))
        self.training_loss_std.append(np.std(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        self.training_loss_min.append(np.min(train_losses))
        self.training_loss_max.append(np.max(train_losses))

        #batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        #batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
        #                  leave=False)
        batch_iter = enumerate(self.validation_DataLoader)

        for i, (x, y) in batch_iter:
            inp, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = torch.squeeze(self.model(inp))  
                
                if (self.criterion_pos==False):
                    loss = self.criterion(out, target) # calculate loss
                else:
                    loss = 1 - self.criterion(out, target)  # calculate loss
                    
                loss_value = loss.item()
                valid_losses.append(loss_value)

                #batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
                # update progressbar every n_steps_progressbar steps
                #if i>0:
                #    if (i%round(len(self.validation_DataLoader)/self.n_steps_progressbar) == 0):
                #        print('\tVal:',i,'/',len(self.validation_DataLoader))
                #else:
                #    print('\tVal:',0,'/',len(self.validation_DataLoader))

        self.validation_loss_mean.append(np.mean(valid_losses))
        self.validation_loss_std.append(np.std(valid_losses))
        self.validation_loss_min.append(np.min(valid_losses))
        self.validation_loss_max.append(np.max(valid_losses))

        #batch_iter.close()
