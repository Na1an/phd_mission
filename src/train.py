import torch
import numpy as np

class Trainer():
    #def __init__(self, model, device, train_data, val_data, opt="Adam"):
    def __init__(self, model, device, opt="Adam"):
        # put our data to device
        self.device = device
        self.model = model.to(device)
        
        # optimizer
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # take data
        #self.train_data = train_data
        #self.val_data = val_data

    def train_model(self, nb_epoch, train_data):
        '''
        Args:
            nb_epoch : a integer. How many epochs you want to train.
            train_data : a 5-D tensor. (B, C, x, y, z)
        Return:
        '''
        # initialize loss
        loss = 0

        for i in range(0, nb_epoch):
            # batch.size = 
            # take n iterations to complete an epoch
            sum_loss = 0
            print("start {}-th epoch".format(i))

        return None