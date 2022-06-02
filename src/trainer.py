import torch
import numpy as np
import torch.nn as nn
from utility import *
from glob import glob
from torch.utils.data import DataLoader
#from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, device, train_dataset, train_voxel_nets, val_dataset, val_voxel_nets, batch_size, sample_size, predict_threshold, num_workers, shuffle=True, opt="Adam"):
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
        self.val_min = None
        self.threshold = predict_threshold
        self.sample_size = sample_size
        self.batch_size = batch_size

        #
        self.writer = SummaryWriter(get_current_direct_path() + "/tensorboard")

    # this is a cute function for calculating the loss
    def compute_loss(points, label, voxel_net):
        loss = 0 
        output = self.model(points, self.train_voxel_nets[voxel_net])
        tmp_loss = nn.functional.binary_cross_entropy_with_logits(output, label)
        return loss

    # let's train it!
    def train_model(self, nb_epoch=200):
        '''
        Args:
            nb_epoch : a integer. How many epochs you want to train.
            train_data : a np.darray. (x, y, z, label)

        Return:
            None.
        '''
        
        print("len(self.train_loader.dataset=", len(self.train_loader.dataset))
        start = self.load_checkpoint()
        for e in range(start, nb_epoch):            
            print('======= Start epoch {} ============='.format(e))
            epoch_loss = 0.0
            epoch_acc = 0.0
            #epoch_num_correct= 0

            if e % 1 == 0:
                self.save_checkpoint(e)
                val_loss, predict_correct = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss 

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.gradient_clipping_path + '/val_min=*'):
                        os.remove(path)
                    np.save(self.gradient_clipping_path + '/val_min={}'.format(e),[e,val_loss])

                print("<<Epoch {}>> - val loss average {} - val accuracy average {}".format(e, val_loss, predict_correct/self.sample_size))
                self.writer.add_scalar('validation loss - avg', val_loss, e)
                self.writer.add_scalar('validation accuracy - avg', predict_correct/self.sample_size, e)

            loader_len = 0
            # points, labels, v_cuboid
            for points, intensity, label, voxel_net in self.train_loader:

                
                self.model.train() # tell torch we are traning
                self.optimizer.zero_grad()

                logits = self.model(points, intensity, self.train_voxel_nets[voxel_net])

                '''
                Visualization model
                ll = make_dot(logits.mean(), params=dict(self.model.named_parameters()))
                ll.view()
                '''
                
                #criterion
                tmp_loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
                tmp_loss.backward()
                self.optimizer.step()

                #preds = logits.argmax(dim=1).float()
                #label_one_dim = label.argmax(dim=1).float()
                num_correct = torch.eq(logits.argmax(dim=1).float(), label.argmax(dim=1).float()).sum().item()/self.batch_size
                #print("num_correct = ", num_correct)
                epoch_loss = epoch_loss + tmp_loss.item()
                epoch_acc = epoch_acc + num_correct/self.sample_size
                loader_len = loader_len + 1
                print("[e={}]>>> [Training] - Current test loss: {} - test accuracy: {}".format(e, tmp_loss.item(), num_correct/self.sample_size))

            print("============ Epoch {}/{} is trained - epoch_loss - {} - epoch_acc - {}===========".format(e, nb_epoch, epoch_loss/loader_len, epoch_acc/loader_len))
            self.writer.add_scalar('training loss - epoch avg', epoch_loss/loader_len, e)
            self.writer.add_scalar('training accuracy - epoch avg', epoch_acc/loader_len, e)

            self.writer.add_scalars('loss train (epoch avg) vs val', 
                {
                    'train_loss': epoch_loss/loader_len,
                    'val_loss': val_loss
                }, e)
            self.writer.add_scalars('acc train (epoche avg) vs val', 
                {
                    'train_acc': epoch_acc/loader_len,
                    'val_acc': predict_correct/self.sample_size
                }, e)

        self.writer.close()

        return None
    
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + '/checkpoint_epoch_{:04}.tar'.format(epoch)
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
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    
    # compute validation/test loss
    def compute_val_loss(self):
        self.model.eval()
        sum_val_loss = 0
        num_batches = 5
        predict_correct = 0
        for _ in range(num_batches):
            #output = self.model(points, self.train_voxel_nets[voxel_net])
            #tmp_loss = nn.functional.binary_cross_entropy_with_logits(output, label)
            try:
                points, label, voxel_net = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_loader.__iter__()
                points, intensity, label, voxel_net = self.val_data_iterator.next()

            logits = self.model(points, intensity, self.train_voxel_nets[voxel_net])
            #logits = output.argmax(dim=1).float()
            
            # loss
            # binary_cross_entropy_with_logits : input doesn't need to be [0,1], but target/label need to be [0, N-1] (therwise the loss will be wired)
            tmp_loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
            sum_val_loss = sum_val_loss + tmp_loss.item()

            # accuracy
            #preds = logits.argmax(dim=1).float()
            num_correct = torch.eq(logits.argmax(dim=1).float(), label.argmax(dim=1).float()).sum().item()/self.batch_size
            predict_correct = predict_correct + num_correct
            
        return sum_val_loss/num_batches, predict_correct/num_batches

'''
single dim output train function
def train_model(self, nb_epoch=200):
        
        Args:
            nb_epoch : a integer. How many epochs you want to train.
            train_data : a np.darray. (x, y, z, label)

        Return:
            None.
        
        
        print("len(self.train_loader.dataset=", len(self.train_loader.dataset))
        start = self.load_checkpoint()
        for e in range(start, nb_epoch):            
            print('======= Start epoch {} ============='.format(e))
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_correct= 0

            if e % 1 == 0:
                self.save_checkpoint(e)
                val_loss, predict_correct = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss 

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.gradient_clipping_path + '/val_min=*'):
                        os.remove(path)
                    np.save(self.gradient_clipping_path + '/val_min={}'.format(e),[e,val_loss])

                print("<<Epoch {}>> - val loss average {} - val accuracy average {}".format(e, val_loss, predict_correct/self.sample_size))

            loader_len = 0
            # points, labels, v_cuboid
            for points, intensity, label, voxel_net in self.train_loader:
                self.model.train() # tell torch we are traning
                self.optimizer.zero_grad()
                # logits.shape [4, 5000]
                logits = self.model(points, intensity, self.train_voxel_nets[voxel_net])
                
                Visualization model
                ll = make_dot(logits.mean(), params=dict(self.model.named_parameters()))
                ll.view()
                
                
                #criterion
                tmp_loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
                tmp_loss.backward()
                self.optimizer.step()

                preds = (logits>0.5).float()
                num_correct = torch.eq(preds, label).sum().item()/self.batch_size
                epoch_loss = epoch_loss + tmp_loss.item()
                loader_len = loader_len + 1
                print("[e={}]>>> [Training] - Current test loss: {} - test accuracy: {}".format(e, tmp_loss.item(), num_correct/self.sample_size))

            print("============ Epoch {}/{} is trained - epoch_loss - {} - e_loss average - {}===========".format(e+1, nb_epoch, epoch_loss, epoch_loss/loader_len))

        return None
'''