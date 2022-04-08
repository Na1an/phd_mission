import torch
import numpy as np

class Trainer():
    #def __init__(self, model, device, train_data, val_data, opt="Adam"):
    def __init__(self, model, device, opt="Adam"):
        # put our data to device
        self.device = device
        self.model = model.to(device)
        
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        # take data
        #self.train_data = train_data
        #self.val_data = val_data

    def train_model(self):
        '''
        Args:
        Return
        '''
        return None