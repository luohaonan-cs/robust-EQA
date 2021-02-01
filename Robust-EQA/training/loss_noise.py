import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models import MaskedNLLCriterion_noise

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    # remember_rate = 1 - forget_rate
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

def loss_coteaching_nav(y_1, y_2, t, forget_rate, masks):
    lossFn = MaskedNLLCriterion_noise().cuda()
    loss_1 = lossFn(y_1, t, masks)

    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = lossFn(y_2, t, masks)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    # remember_rate = 1 - forget_rate
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = lossFn(y_1[ind_2_update], t[ind_2_update], masks[:num_remember])
    loss_2_update = lossFn(y_2[ind_1_update], t[ind_1_update], masks[:num_remember])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

