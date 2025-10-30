import pdb
import copy
import utils
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer
import modules.SkeletonNet as SkeletonNet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, hidden_size=1280, gloss_dict=None, input_type='keypoint', loss_weights=None,
            weight_norm=True, cfg=None,
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.input_type = input_type
        self.visual_backbone_keypoint = SkeletonNet.CESMM(num_channel=3, input_type=self.input_type)

        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)

        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)

    def forward(self, len_x, label=None, label_lgt=None, keypoint=None):    
        keypoint = self.visual_backbone_keypoint(keypoint).permute(2,0,1)  #B,C,T -> T,B,C
        conv1d_outputs = self.classifier(keypoint)
        lgt = len_x.cpu()
        tm_outputs = self.temporal_model(keypoint, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs, lgt, batch_first=False, probs=False)

        return {
            "feat_len": lgt,
            "conv_logits": conv1d_outputs,
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['ConvCTC']
            elif k == 'SeqCTC':
                total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['SeqCTC']
            elif k == 'Dist':
                total_loss['Dist'] = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
                loss += total_loss['Dist']
                total_loss['Dist'] = weight * self.loss['distillation'](ret_dict["sequence_logits"],
                                                           ret_dict["conv_logits"].detach(),
                                                           use_blank=False)
                loss += total_loss['Dist']
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss