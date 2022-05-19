import torch
import numpy as np
import torch.nn as nn
from utility import *
from glob import glob
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, model, device, train_dataset, train_voxel_nets, val_dataset, val_voxel_nets, batch_size, num_workers, shuffle=True, opt="Adam"):
        '''
        Args:
            model: the Deep Learning model.
            train_dataset: the dataset we need.
            voxel_skeleton: 
            batch_size: a integer.
            num_workers: a integer.
            shuffle: shuffler or not.
            opt: optimizer.
        '''
        # put our data to device & DataLoader
        self.device = device
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.train_voxel_nets = torch.from_numpy(train_voxel_nets.copy()).type(torch.float).to(self.device)

        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_voxel_nets = torch.from_numpy(val_voxel_nets.copy()).type(torch.float).to(self.device)
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # check_point path
        self.checkpoint_path = get_current_direct_path() + "/checkpoints"
        if not os.path.exists(self.checkpoint_path):
            print(">> No checkpoint folder exist, so create it:", self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        else:
            print(">> So checkpoint folder exist, the path is:", self.checkpoint_path)

        self.gradient_clipping_path = get_current_direct_path() + "/gradient_clipping"
        if not os.path.exists(self.gradient_clipping_path):
            print(">> No gradient clipping folder exist, so create it:", self.checkpoint_path)
            os.makedirs(self.gradient_clipping_path)
        else:
            print(">> So gradient clipping folder exist, the path is:", self.checkpoint_path)
        # gradient_clipping
        self.val_loss = None

    def train_model(self, nb_epoch=200):
        '''
        Args:
            nb_epoch : a integer. How many epochs you want to train.
            train_data : a np.darray. (x, y, z, label)

        Return:
            None.
        '''
        epoch_loss=0 # 为了compute_loss()
        #epoch_acc=0
        start = self.load_checkpoint()

        for e in range(start, nb_epoch):
            self.model.train() # tell torch we are traning
            print('======= Start epoch {} ============='.format(e))
            epoch_loss = 0.0

            if e % 1 == 0:
                self.save_checkpoint(e)
                val_loss = self.compute_val_loss()

                if self.val_loss is None:
                    self.val_loss = val_loss 

                if val_loss < self.val_loss:
                    self.val_loss = val_loss
                    for path in glob(self.gradient_clipping_path + '/val_loss=*'):
                        os.remove(path)
                    np.save(self.gradient_clipping_path + '/val_loss={}'.format(e),[e,val_loss])
                
            # points, labels, v_cuboid
            for points, label, voxel_net in self.train_loader:
                
                self.optimizer.zero_grad()
                output = self.model(points, self.train_voxel_nets[voxel_net])

                #criterion
                tmp_loss = nn.functional.binary_cross_entropy_with_logits(output, label)

                tmp_loss.backward()
                self.optimizer.step()
                print(">>>> Current loss: {}".format(tmp_loss.item()))
                epoch_loss = epoch_loss + tmp_loss.item()

            print("============ Epoch {}/{} is trained - epoch_loss - {} - e_loss average - {}===========".format(e, nb_epoch, epoch_loss/nb_epoch))

        return None
    
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + '/checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch':epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, 
                        path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + '/checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    
    # compute validation/test loss
    def compute_val_loss(self):
        self.model.eval()
        sum_val_loss = 0
        num_batches = 5
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss(val_batch).item()

        return sum_val_loss/num_batches
