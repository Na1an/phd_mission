import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utility import *
from glob import glob
from torch.utils.data import DataLoader
#from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from captum.attr import LayerGradCam, Saliency, LayerActivation
from sklearn.utils import class_weight

class Trainer():
    def __init__(self, model, device, train_dataset, train_voxel_nets, val_dataset, val_voxel_nets, batch_size, sample_size, predict_threshold, num_workers, grid_size, global_height, shuffle=True, opt="Adam"):
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=0.0001)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9, nesterov=True)

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
    '''
    def compute_loss(points, label, voxel_net):
        loss = 0 
        output = self.model(points, self.train_voxel_nets[voxel_net])
        tmp_loss = nn.functional.binary_cross_entropy_with_logits(output, label)
        return loss
    '''

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
            epoch_auroc = 0.0
            loader_len = 1
            # points, labels, v_cuboid
            for points, pointwise_features, label, voxel_net in self.train_loader:
                
                self.model.train() # tell torch we are traning
                self.optimizer.zero_grad()
                
                
                points_for_pointnet = torch.cat([points.transpose(2,1), pointwise_features.transpose(2,1), points.transpose(2,1)], dim=1)
                #points_for_pointnet[:,:2,:] = points_for_pointnet[:,0:2,:] * self.grid_size
                #points_for_pointnet[:,2,:] = points_for_pointnet[:,2,:] * self.global_height + (self.global_height/2)
                points_for_pointnet[:,6:,:] = points_for_pointnet[:,6:,:] + 0.5
                print("pointsfor_pointnet.shape={}".format(points_for_pointnet.shape))
                
                #print(">>> points.shape = {}, pointwise_features.shape={}, labels.shape={}, voxel_net.shape={}".format(points.shape, pointwise_features.shape, label.shape, voxel_net.shape))
                logits = self.model(points, pointwise_features, voxel_net, points_for_pointnet)
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
                y_true = label.detach().clone().cpu().data.numpy().transpose(0,2,1).reshape(self.batch_size*self.sample_size, 2).astype('int64')
                logits = logits.permute(0,2,1).reshape(self.batch_size*self.sample_size, 2).to(self.device)
                logits = F.softmax(logits, dim=1)
                logits_wood = logits[:,1].detach().clone().cpu().data.numpy()
                label = label.permute(0,2,1).reshape(self.batch_size*self.sample_size, 2).to(self.device)
                
                class_weights=class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(np.argmax(y_true, axis=1)), y=np.argmax(y_true, axis=1))
                class_weights=torch.tensor(class_weights, dtype=torch.float)
                
                tmp_loss = nn.functional.binary_cross_entropy_with_logits(weight=class_weights.to(self.device), input=logits, target=label)
               
                '''
                print(">>>> [new] logits.shape = {}, label.shape = {}".format(logits.shape, label.shape))
                print(">>>> [new] logits = {}, label = {}".format(logits[0:5], label[0:5]))
                '''

                tmp_loss.backward()
                self.optimizer.step()
                
                #res1, res2 = label.max(1), res = [0.77, 0.78, 0.22, ... proba] res2 = [1,1,0,0... label]
                _, label = label.max(1)
                _, logits = logits.max(1)

                num_correct = torch.eq(logits.to(self.device), label.to(self.device)).sum().item()
                
                #print(" logits.argmax(dim=1).float() shape = {} label.argmax(dim=1).float() shape = {} num_correct = {}".format(logits.float().shape, label.float().shape,num_correct))
                #logits = logits.reshape(self.batch_size*self.sample_size)
                #label = label.reshape(self.batch_size*self.sample_size)
                #print("logits.shape = {} logits = {}\n, label.shape = {} label = {}\n".format(logits.shape, logits, label.shape, label))
                
                print("bincount logits.shape={}".format(torch.bincount(logits)))
                print("bincount label.shape={}".format(torch.bincount(label)))
                
                # [metric bloc] precision, recall, f1-score
                #auroc = calculate_auroc(y_score=logits[:,1], y_true=label)
                
                label = label.detach().clone().cpu().data.numpy()
                logits = logits.detach().clone().cpu().data.numpy()

                #auroc_score = roc_auc_score(y_score=logits_wood, y_true=label)
                cf_matrix = confusion_matrix(label, logits, labels=[0,1])
                tn, fp, fn, tp = cf_matrix.ravel()
                recall, specificity, precision, npv, fpr, fnr, fdr, acc = calculate_recall_precision(tn, fp, fn, tp)
                m_spe = BinarySpecificity()
                specificity = m_spe(logits, label)
                f1_score_val = f1_score(label, logits)
                #print("tn-{} fp-{} fn-{} tp-{} recall-{} specificity-{} precision-{} npv-{} fpr-{} fnr-{} fdr-{} acc-{} f1_score-{}".format(tn, fp, fn, tp, recall, specificity, precision, npv, fpr, fnr, fdr, acc, f1_score_val))

                epoch_loss = epoch_loss + tmp_loss.item()
                epoch_acc = epoch_acc + num_correct/self.sample_size
                epoch_specificity = epoch_specificity + specificity
                #epoch_auroc = epoch_auroc + auroc_score
                loader_len = loader_len + 1
                print("[e={}]>>> [Training] - Current test loss: {} - accuracy - {} specificity - {}".format(e, tmp_loss.item(), num_correct/(self.sample_size*self.batch_size), specificity))
            
            print("============ Epoch {}/{} is trained - epoch_loss - {} - epoch_acc - {} epoch_specificity - {}===========".format(e, nb_epoch, epoch_loss/loader_len, epoch_acc/(loader_len*self.batch_size), epoch_specificity/loader_len))
            self.writer.add_scalar('training loss - epoch avg', epoch_loss/loader_len, e)
            self.writer.add_scalar('training accuracy - epoch avg', epoch_acc/(loader_len*self.batch_size), e)
            self.writer.add_scalar('training specificity - epoch avg', epoch_specificity/(loader_len), e)
            self.writer.add_scalar('training auroc - epoch avg', epoch_auroc/(loader_len), e)

            if e % 1 == 0:
                self.save_checkpoint(e)
                val_loss, predict_correct, mcc, df_cm, list_stat_res = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.gradient_clipping_path + '/val_min=*'):
                        os.remove(path)
                    np.save(self.gradient_clipping_path + '/val_min={}'.format(e),[e,val_loss])
                    #print(">> val_min saved here :",self.gradient_clipping_path,"val_min=".format(e))
                
                print("<<Epoch {}>> - val loss average {} - val accuracy average {}".format(e, val_loss, predict_correct/self.sample_size))
                self.writer.add_scalar('validation loss - avg', val_loss, e)
                self.writer.add_scalar('validation accuracy - avg', predict_correct/self.sample_size, e)
                # add matthew correlation coefficient
                #list_stat_res = [tn, fp, fn, tp, recall, specificity, precision, npv, fpr, fnr, fdr, acc, f1_score_val, auroc]
                self.writer.add_scalar('tn - avg', list_stat_res[0], e)
                self.writer.add_scalar('fp - avg', list_stat_res[1], e)
                self.writer.add_scalar('fn - avg', list_stat_res[2], e)
                self.writer.add_scalar('tp - avg', list_stat_res[3], e)
                self.writer.add_scalar('recall - avg', list_stat_res[4], e)
                self.writer.add_scalar('specificity - avg', list_stat_res[5], e)
                self.writer.add_scalar('precision - avg', list_stat_res[6], e)
                self.writer.add_scalar('npv - avg', list_stat_res[7], e)
                self.writer.add_scalar('fpr - avg', list_stat_res[8], e)
                self.writer.add_scalar('fnr - avg', list_stat_res[9], e)
                self.writer.add_scalar('fdr - avg', list_stat_res[10], e)
                self.writer.add_scalar('f1_score - avg', list_stat_res[12], e)
                self.writer.add_scalar('auroc - avg', list_stat_res[13], e)
                
                #fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
                #plt.close(fig_)
                #self.writer.add_figure("Confusion matrix", fig, e)

            self.writer.add_scalars('Loss', 
                {
                    'train_loss (epoch average)': epoch_loss/loader_len,
                    'val_loss': val_loss
                }, e)
            self.writer.add_scalars('Accuracy', 
                {
                    'train_acc (epoch average)': epoch_acc/(loader_len*self.batch_size),
                    'val_acc': predict_correct/self.sample_size
                }, e)
            self.writer.add_scalars('Specificity', 
                {
                    'train_sp (epoch average)': epoch_specificity/loader_len,
                    'val_sp': list_stat_res[5] # validation specificity
                }, e)

        self.writer.close()
        return None
    
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + '/checkpoint_epoch_{:04}.pth'.format(epoch)
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
        path = self.checkpoint_path + '/checkpoint_epoch_{:04}.pth'.format(checkpoints[-1])

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
        
        mcc = 0
        y_true_all = np.zeros((num_batches, self.sample_size*self.batch_size))
        y_predict_all = np.zeros((num_batches, self.sample_size*self.batch_size))
        y_predict_wood_all = np.zeros((num_batches, self.sample_size*self.batch_size))
        for nb in range(num_batches):
            #output = self.model(points, self.train_voxel_nets[voxel_net])
            #tmp_loss = nn.functional.binary_cross_entropy_with_logits(output, label)
            try:
                points, label, voxel_net = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_loader.__iter__()
                points, pointwise_features, label, voxel_net = self.val_data_iterator.next()
            
            points_for_pointnet = torch.cat([points.transpose(2,1), pointwise_features.transpose(2,1), points.transpose(2,1)], dim=1)
            #points_for_pointnet[:,0:2,:] = points_for_pointnet[:,0:2,:] * self.grid_size
            #points_for_pointnet[:,2,:] = points_for_pointnet[:,2,:] * self.global_height + (self.global_height/2)
            points_for_pointnet[:,6:,:] = points_for_pointnet[:,6:,:] + 0.5
            #print("pointsfor_pointnet.shape={}".format(points_for_pointnet.shape))
            logits = self.model(points, pointwise_features, voxel_net, points_for_pointnet)
            logits = logits.permute(0,2,1).reshape(self.batch_size*self.sample_size, 2)
            logits = F.softmax(logits, dim=1)
            label = label.permute(0,2,1).reshape(self.batch_size*self.sample_size, 2)

            #logits = output.argmax(dim=1).float()
            #preds, answer_id = nn.functional.softmax(logits, dim=1).data.cpu().max(dim=1)
            
            # version laptop
            #y_true = label.detach().numpy().transpose(0,2,1).reshape(self.batch_size*self.sample_size, 2).astype('int64')
            #y_predict = logits.detach().numpy().transpose(0,2,1).reshape(self.batch_size*self.sample_size, 2).astype('int64')
            
            # version cluster
            #y_true = label.detach().numpy().astype('int64')
            #y_predict = logits.detach().numpy().astype('float64')
            #y_predict = F.softmax(y_predict, dim=1)

            y_true_all[nb] = np.argmax(label.detach().numpy().astype('int64'), axis=1)
            y_predict_all[nb] = np.argmax(logits.detach().numpy().astype('float64'), axis=1)
            print("shape: y_true={}, y_predict={}".format( y_true_all[nb].shape, y_predict_all[nb].shape))

            '''
            # evaluate the contribution of layers
            layer_act = LayerActivation(self.model, self.model.fc_0)
            attribution = layer_act.attribute(points, intensity, self.train_voxel_nets[voxel_net], target=answer_id)
            print("attribution=", attribution)
            plt.plot(attribution.to('cpu').numpy(),output.to('cpu').numpy())
            '''
            '''
            layer_gc = LayerGradCam(self.model, self.model.fc_0)
            vv = self.train_voxel_nets[voxel_net]
            attr0 = layer_gc.attribute(points, intensity, vv, target=1)
            #attr1 = layer_gc.attribute(points, intensity, self.train_voxel_nets[voxel_net])
            print("atttr0=", attr0)
            #print("atttr1=", attr1)
            plt.plot(attr0.to('cpu').numpy(),output.to('cpu').numpy())
            #plt.plot(attr1.to('cpu').numpy(),output.to('cpu').numpy())
            '''
            '''
            saliency = Saliency(self.model)
            attribution = saliency.attribute(points, intensity, self.train_voxel_nets[voxel_net],target=0)
            '''

            # loss
            # binary_cross_entropy_with_logits : input doesn't need to be [0,1], but target/label need to be [0, N-1] (therwise the loss will be wired)
            #tmp_loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
            class_weights=class_weight.compute_class_weight(class_weight="balanced", classes=[0,1], y=y_true_all[nb])
            class_weights=torch.tensor(class_weights, dtype=torch.float)
            print("[val]>>> class_weights = {}".format(class_weights))
            # with weights
            tmp_loss = nn.functional.binary_cross_entropy_with_logits(weight=class_weights.to(self.device), reduction='mean', input=logits, target=label)
            
            #tmp_loss = nn.functional.binary_cross_entropy_with_logits(reduction='mean', input=logits.to(self.device), target=label.to(self.device))
            sum_val_loss = sum_val_loss + tmp_loss.item()

            # accuracy
            #preds = logits.argmax(dim=1).float()
            num_correct = torch.eq(logits.argmax(dim=1).float(),label.argmax(dim=1).float()).sum().item()/self.batch_size
            predict_correct = predict_correct + num_correct
        
        print("y_true_all.shape={} y_true_all = {}".format(y_true_all.shape, y_true_all))
        print("y_predict_all.shape={} y_predict_all = {}".format(y_predict_all.shape, y_predict_all))
        #print("bincount y_true_all.shape={}".format(np.bincount(y_true_all)))
        #print("bincount y_predict_all.shape={}".format(np.bincount(y_predict_all)))
        
        y_true_all = y_true_all.reshape(num_batches*self.sample_size*self.batch_size).astype(int)
        y_predict_all = y_predict_all.reshape(num_batches*self.sample_size*self.batch_size).astype(int)
        y_predict_wood_all = y_predict_wood_all.reshape(num_batches*self.sample_size*self.batch_size).astype(int)
        #print("shape: y_true={}, y_predict={}".format(y_true_all.shape, y_predict_all.shape))
        print("bincount y_true_all.shape={}".format(np.bincount(y_true_all)))
        print("bincount y_predict_all.shape={}".format(np.bincount(y_predict_all)))
        #cf_matrix = confusion_matrix(y_true_all, y_predict_all, labels=[0,1])
        
        mcc = matthews_corrcoef(y_true_all, y_predict_all)
        classes = ('leaf', 'wood')
        cf_matrix = confusion_matrix(y_true_all, y_predict_all, labels=[0,1])
        
        auroc_score = roc_auc_score(y_score=y_predict_wood_all, y_true=y_true_all)

        tn, fp, fn, tp = cf_matrix.ravel()
        print("tn-{} fp-{} fn-{} tp-{}".format(tn, fp, fn, tp))
        # precision, recall, f1-score
        recall, specificity, precision, npv, fpr, fnr, fdr, acc = calculate_recall_precision(tn, fp, fn, tp)
        m_spe = BinarySpecificity()
        specificity = m_spe(y_predict_wood_all, y_true_all)
        f1_score_val = f1_score(y_true_all, y_predict_all)
        print("tn-{} fp-{} fn-{} tp-{} recall-{} specificity-{} precision-{} npv-{} fpr-{} fnr-{} fdr-{} acc-{} f1_score-{} auroc-{}".format(tn, fp, fn, tp, recall, specificity, precision, npv, fpr, fnr, fdr, acc, f1_score_val, auroc_score))
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [ic for ic in classes], columns = [ic for ic in classes])

        list_stat_res = [tn, fp, fn, tp, recall, specificity, precision, npv, fpr, fnr, fdr, acc, f1_score_val, auroc_score]

        return sum_val_loss/num_batches, predict_correct/num_batches, mcc, df_cm, list_stat_res

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
