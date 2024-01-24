import torch
import torch.nn as nn
import pickle
import pytorch_lightning as pl
from torch import nn, optim
import timm
import torchmetrics
from torch.optim import Adam
import pandas
import config

class Encoder(pl.LightningModule):
    def __init__(self, model_name, emb_dim):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained = False)
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.in_features, emb_dim)
        self.loss_fn = SupConLoss(0.07, 'one', 0.07)


    def forward(self, x):
        emb = self.backbone(x)
        return emb

    # Difference between Normal and Lightning: The train, valid and test steps is written here inside the class
    def training_step(self, batch, batch_idx):
        images, labels = batch

        bsz = len(labels)

        images = torch.cat([images[0], images[1]], dim=0)

        features = self.forward(images)

        # Manipulating the features for SupConLoss
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Calculating SupConLoss
        loss = self.loss_fn(features,labels)

        return loss

    def validation_step(self, batch , batch_idx):

        images, labels = batch

        bsz = len(labels)

        images = torch.cat([images[0], images[1]], dim=0)
        
        #print(images.shape)

        features = self.forward(images)

        # Manipulating the arrangment of features for SupConLoss
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Calculating SupConLoss
        loss = self.loss_fn(features,labels)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        
        bsz = len(labels)

        images = torch.cat([images[0], images[1]], dim=0)
        

        features = self.forward(images)

        # Manipulating the arrangment of features for SupConLoss
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Calculating SupConLoss
        loss = self.loss_fn(features,labels)
        return loss

    # We can add schedulers to this method
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 0.001)
    

# SupConLoss: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConCE(pl.LightningModule):
    def __init__(self, train=True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = 264)
        self.f1_score = torchmetrics.F1Score(task = 'multiclass', num_classes = 264)
        backbone = 'resnet50'
        model_path = config.encoder_path
        pretrained_model = Encoder.load_from_checkpoint(model_path, model_name = backbone, emb_dim = 128)


        #Freezing all the encoder layers
        for param in pretrained_model.parameters():
            param.requires_grad = False


        #Trainging only the last layer
        pretrained_model.backbone.fc = nn.Linear(in_features=pretrained_model.backbone.fc.in_features, out_features=264)

        pretrained_model.backbone.fc.requires_grad = True

        self.model = pretrained_model


    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_pred = self.forward(images)
        loss = self.loss_fn(y_pred,labels)
        accuracy = self.accuracy(y_pred,labels)
        f1_score = self.f1_score(y_pred,labels)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score},
                      on_step = False, on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_pred = self.forward(images)
        loss = self.loss_fn(y_pred,labels)
        accuracy = self.accuracy(y_pred,labels)
        f1_score = self.f1_score(y_pred,labels)
        
        #one_hot_target = F.one_hot(labels, num_classes=264)
        
        #y_pred = pd.DataFrame(y_pred.cpu().detach().numpy())
        #y_true = pd.DataFrame(one_hot_target.cpu().detach().numpy())
        
        #cmap_score = padded_cmap(y_true, y_pred)
        
        self.log_dict({'valid_loss': loss, 'valid_accuracy': accuracy, 'valid_f1_score': f1_score, 'cmap_score': cmap_score},
                      on_step = False, on_epoch = True, prog_bar = True)
        return loss
        

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 0.001)