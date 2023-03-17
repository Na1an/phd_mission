import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utility import *
from glob import glob
from torch.utils.data import DataLoader
#from torchviz import make_dot
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from captum.attr import LayerGradCam, Saliency, LayerActivation

class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        # leave label=1, wood label=0
        # 提高
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':    
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

# greg loss
def make_weights_for_celoss(target):
    n,l,p = target.shape
    
    # flatten the target_tmp
    target_tmp = target[:,1,:].reshape(n*1*p)
    
    wood_loss = (target_tmp < 0.5).nonzero(as_tuple=True)[0]
    leaf_loss = (target_tmp > 0.5).nonzero(as_tuple=True)[0]
    res = torch.zeros_like(target_tmp)
    index_leaf = 0
    if len(wood_loss) < len(leaf_loss):
        index_leaf = torch.randperm(len(leaf_loss))[:len(wood_loss)]
        leaf_loss = leaf_loss[index_leaf]
    #print("index_leaf={} leaf_loss={} wood_loss={}".format(index_leaf, leaf_loss, wood_loss))
    res[wood_loss] = 0.5 / len(wood_loss)
    res[leaf_loss] = 0.5 / len(leaf_loss)
    res = res.reshape(n,1,p)
    torch.cat((res,res), axis=1)
    #print("res.shape=", res.shape)
    return torch.cat((res,res), axis=1)/2

class Trainer():
    def __init__(self, model, device, train_dataset, train_voxel_nets, val_dataset, val_voxel_nets, batch_size, sample_size, predict_threshold, num_workers, grid_size, global_height, alpha=0, gamma=2, shuffle=True, opt="Adam"):
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
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        #self.train_voxel_nets = torch.from_numpy(train_voxel_nets.copy()).type(torch.float).to(self.device)

        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #self.val_voxel_nets = torch.from_numpy(val_voxel_nets.copy()).type(torch.float).to(self.device)
        self.grid_size = grid_size
        self.global_height = global_height

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-7, weight_decay=0)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=2e-3, momentum=0.9, nesterov=True)

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

        # alpha is for label=1, (1-alpha) if for label=0
        # so in our case, if wood label=0, we should make alpha=0.2 (leaf label=1) for example -> make leave points less important
        # but above is only true for gamma=0
        # if gamma>0, alpha is not only a weight for adjusting the diff weights for both calsses
        # also, alpha is used to adjust the big loss value bring by gamma
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = FocalLoss(gamma=self.gamma,alpha=self.alpha)
        self.writer = SummaryWriter(get_current_direct_path() + "/tensorboard")


    # let's train it!
    def train_model(self, nb_epoch=200):
        '''
        Args:
            nb_epoch : a integer. How many epochs you want to train.
            train_data : a np.darray. (x, y, z, label)

        Return:
            None.
        '''

        print("len(self.train_loader.dataset) =", len(self.train_loader.dataset))
        start = self.load_checkpoint()
        for e in range(start, nb_epoch):         
            print('======= Start epoch {} ============='.format(e))
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_specificity = 0.0
            epoch_recall = 0.0
            epoch_auroc = 0.0
            loader_len = 1
            # points, labels, v_cuboid
            for points, pointwise_features, label, voxel_net, points_raw in self.train_loader:
                
                self.model.train() # tell torch we are traning
                self.optimizer.zero_grad()
                
                points_for_pointnet = torch.cat([points.transpose(2,1), pointwise_features.transpose(2,1)], dim=1)
                #print(">>> points_for_pointnet.shape = {}, pointwise_features.shape={}".format(points_for_pointnet.shape, pointwise_features.shape))
                #print(">>> points.shape = {}, pointwise_features.shape={}, labels.shape={}, voxel_net.shape={}".format(points.shape, pointwise_features.shape, label.shape, voxel_net.shape))
                logits = self.model(points, pointwise_features, voxel_net, points_for_pointnet.float())
                #print("logits.shape = {}, logits[0:10]={}".format(logits.shape,logits[0:10]))

                '''
                Visualization model
                ll = make_dot(logits.mean(), params=dict(self.model.named_parameters()))
                ll.view()
                '''
                
                #criterion
                # version laptop
                #y_true = label.detach().numpy().transpose(0,2,1).reshape(self.batch_size*self.sample_size, 2).astype('int64')
                
                # version cluster
                #print("label.shape={}".format(label.shape))
                
                logits = logits.to(self.device)
                label = label.to(self.device)
                #tmp_loss = self.criterion(logits, label)

                ce_loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
                index_loss = make_weights_for_celoss(label)
                ce_loss.backward(index_loss)
                self.optimizer.step()
                
                #res1, res2 = label.max(1), res = [0.77, 0.78, 0.22, ... proba] res2 = [1,1,0,0... label]
                _, label = label.max(1)
                _, logits = logits.max(1)

                #logits = torch.where(logits>0.99, 1, 0)
                #print("logits.shape={}, label.shape={}".format(logits.shape, label.shape))
                #print("logits[:,0:5]={}, label[:,0:5].shape={}".format(logits[:5], label[:,0:5]))
                num_correct = torch.eq(logits.to(self.device), label.to(self.device)).sum().item()
                
                #print(" logits.argmax(dim=1).float() shape = {} label.argmax(dim=1).float() shape = {} num_correct = {}".format(logits.float().shape, label.float().shape,num_correct))
                logits = logits.reshape(self.batch_size*self.sample_size)
                label = label.reshape(self.batch_size*self.sample_size)
                #print("logits.shape = {} logits = {}\n, label.shape = {} label = {}\n".format(logits.shape, logits, label.shape, label))
                
                print("bincount logits.shape={}".format(torch.bincount(logits)))
                print("bincount label.shape={}".format(torch.bincount(label)))
                
                # [metric bloc] precision, recall, f1-score
                #auroc = calculate_auroc(y_score=logits[:,1], y_true=label)
                
                label = label.detach().clone().cpu().data.numpy()
                logits = logits.detach().clone().cpu().data.numpy()

                #auroc_score = roc_auc_score(y_score=logits_wood, y_true=label)
                cf_matrix = confusion_matrix(y_true=label, y_pred=logits, labels=[0,1])
                tn, fp, fn, tp = cf_matrix.ravel()
                recall, specificity, precision, acc = calculate_recall_precision(tn, fp, fn, tp)
                
                if index_loss.sum() == 0:
                    tmp_loss = ce_loss.mean()
                else:
                    tmp_loss = ce_loss*index_loss
                epoch_loss = epoch_loss + tmp_loss.sum().item()

                epoch_acc = epoch_acc + num_correct/self.sample_size
                epoch_specificity = epoch_specificity + specificity
                epoch_recall = epoch_recall + recall
                #epoch_auroc = epoch_auroc + auroc_score
                loader_len = loader_len + 1
                print("[e={}]>>> [Training] - Current test loss: {} - accuracy - {} specificity - {} recall - {}".format(e, tmp_loss.sum().item(), num_correct/(self.sample_size*self.batch_size), specificity, recall))
            
            print("============ Epoch {}/{} is trained - epoch_loss - {} - epoch_acc - {} epoch_specificity - {} epoch_recall - {}===========".format(e, nb_epoch, epoch_loss/loader_len, epoch_acc/(loader_len*self.batch_size), epoch_specificity/loader_len, epoch_recall/loader_len))
            self.writer.add_scalar('training loss - epoch avg', epoch_loss/loader_len, e)
            self.writer.add_scalar('training accuracy - epoch avg', epoch_acc/(loader_len*self.batch_size), e)
            self.writer.add_scalar('training specificity - epoch avg', epoch_specificity/(loader_len), e)
            self.writer.add_scalar('training auroc - epoch avg', epoch_auroc/(loader_len), e)
            self.writer.add_scalar('training recall - epoch avg', epoch_recall/(loader_len), e)

            if e % 1 == 0:
                self.save_checkpoint(e)
                val_loss, val_acc, mcc, list_stat_res = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.gradient_clipping_path + '/val_min=*'):
                        os.remove(path)
                    np.save(self.gradient_clipping_path + '/val_min={}'.format(e),[e,val_loss])
                    #print(">> val_min saved here :",self.gradient_clipping_path,"val_min=".format(e))
                
                print("<<Epoch {}>> - val loss average {} - val accuracy average {}".format(e, val_loss, val_acc))
                self.writer.add_scalar('validation loss - avg', val_loss, e)
                self.writer.add_scalar('validation accuracy - avg', val_acc, e)
                # add matthew correlation coefficient
                #list_stat_res = [recall, specificity, precision, acc, f1_score, auroc, mcc]
                self.writer.add_scalar('recall - avg', list_stat_res[0], e)
                self.writer.add_scalar('specificity - avg', list_stat_res[1], e)
                self.writer.add_scalar('precision - avg', list_stat_res[2], e)
                self.writer.add_scalar('accuracy - avg', list_stat_res[3], e)
                self.writer.add_scalar('f1_score - avg', list_stat_res[4], e)
                self.writer.add_scalar('auroc - avg', list_stat_res[5], e)
                self.writer.add_scalar('mcc - avg', list_stat_res[6], e)

            self.writer.add_scalars('Loss', 
                {
                    'train_loss (epoch average)': epoch_loss/loader_len,
                    'val_loss': val_loss
                }, e)
            self.writer.add_scalars('Accuracy', 
                {
                    'train_acc (epoch average)': epoch_acc/(loader_len*self.batch_size),
                    'val_acc': val_acc
                }, e)
            self.writer.add_scalars('Specificity', 
                {
                    'train_sp (epoch average)': epoch_specificity/loader_len,
                    'val_sp': list_stat_res[1] # validation specificity
                }, e)
            self.writer.add_scalars('Recall', 
                {
                    'train_recall (epoch average)': epoch_recall/loader_len,
                    'val_recall': list_stat_res[0] # validation recall
                }, e)

        self.writer.close()
        return None
    
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + '/checkpoint_epoch_{:06}.pth'.format(epoch)
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
        path = self.checkpoint_path + '/checkpoint_epoch_{:06}.pth'.format(checkpoints[-1])

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
        mcc, f1_score_all = 0,0
        rec_all, spe_all, pre_all, acc_all, f1_all, auroc_all = [],[],[],[],[],[]
        
        for nb in range(num_batches):
            #output = self.model(points, self.train_voxel_nets[voxel_net])
            #tmp_loss = nn.functional.binary_cross_entropy_with_logits(output, label)
            try:
                points, label, voxel_net = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_loader.__iter__()
                points, pointwise_features, label, voxel_net, points_raw = self.val_data_iterator.next()
            
            points_for_pointnet = torch.cat([points.transpose(2,1), pointwise_features.transpose(2,1)], dim=1)
            logits = self.model(points, pointwise_features, voxel_net, points_for_pointnet.float())
            #logits = logits.permute(0,2,1).reshape(self.batch_size*self.sample_size, 2)
            #logits = F.softmax(logits, dim=1)
            #label = label.permute(0,2,1).reshape(self.batch_size*self.sample_size, 2)

            logits = logits.to(self.device)
            label = label.to(self.device)

            # loss
            #tmp_loss = self.criterion(logits, label)
            ce_loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
            index_loss = make_weights_for_celoss(label)
            if index_loss.sum() == 0:
                tmp_loss = ce_loss.mean()
            else:
                tmp_loss = ce_loss*index_loss
            sum_val_loss = sum_val_loss + tmp_loss.sum().item()

            # accuracy
            #preds = logits.argmax(dim=1).float()
            _,logits = logits.max(1)
            _,label = label.max(1)

            num_correct = torch.eq(logits,label).sum().item()/self.batch_size
            predict_correct = predict_correct + num_correct
            
            logits = logits.reshape(self.batch_size*self.sample_size)
            label = label.reshape(self.batch_size*self.sample_size)
            print("bincount y_true.shape={}".format(torch.bincount(label)))
            print("bincount y_predict.shape={}".format(torch.bincount(logits)))

            logits = logits.detach().clone().cpu().data.numpy()
            label = label.detach().clone().cpu().data.numpy()

            # tn tp fn tp
            tn, fp, fn, tp = confusion_matrix(label, logits, labels=[0,1]).ravel()
            print("tn-{} fp-{} fn-{} tp-{}".format(tn, fp, fn, tp))
            # precision, recall, f1-score
            recall, specificity, precision, acc = calculate_recall_precision(tn, fp, fn, tp)
            rec_all.append(recall)
            pre_all.append(precision)
            spe_all.append(specificity)
            acc_all.append(acc)
            try:
                auroc_all.append(roc_auc_score(y_true=label, y_score=logits))
            except ValueError:
                auroc_all.append(np.nan)
            
            # mcc, auroc_score, f1_score
            mcc = mcc + matthews_corrcoef(y_true=label, y_pred=logits)
            f1_score_all = f1_score_all + f1_score(y_true=label, y_pred=logits, average='macro')
            
        mcc = mcc/num_batches
        auroc_all = np.array(auroc_all).astype(np.float64)
        auroc = np.mean(auroc_all[~np.isnan(auroc_all)])
        f1_score_avg = f1_score_all/num_batches
        
        print("recall-{} specificity-{} precision-{} acc-{} f1_score-{} auroc-{} mcc-{}".format(np.mean(rec_all), np.mean(spe_all), np.mean(pre_all), np.mean(acc_all), f1_score_avg, auroc, mcc))
        list_stat_res = [np.mean(rec_all), np.mean(spe_all), np.mean(pre_all), np.mean(acc_all), f1_score_avg, auroc, mcc]

        # return [validation loss, validation accuracy, mcc, list of metrics]
        return sum_val_loss/num_batches, predict_correct/(num_batches*self.sample_size), mcc, list_stat_res

