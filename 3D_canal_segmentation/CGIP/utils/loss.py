import numpy as np
import torch
import torch.nn as nn


def _sigmoid(x, e=1e-4):
    return x.sigmoid_()
    #return torch.clamp(x.sigmoid_(), min=e, max=1-e)


def get_SR_MSE(pts):
    sr_term = 0

    # add (d_before - d_next)**2 for all tooth, except wisdom tooth
    for i in range(32):
        if i % 8 > 5:
            continue

        p_this = (pts[:, 2*i], pts[:, 2*i + 1])
        p_next = (pts[:, 2*i + 2], pts[:, 2*i + 3])
        if i % 8 == 0:
            if i // 8 == 0:
                p_before = (pts[:, 16], pts[:, 17])
            elif i // 8 == 1:
                p_before = (pts[:, 0], pts[:, 1])
            elif i // 8 == 2:
                p_before = (pts[:, 48], pts[:, 49])
            else:
                p_before = (pts[:, 32], pts[:, 33])
        else:
            p_before = (pts[:, 2*i - 2], pts[:, 2*i - 1])

        d_before = ((p_this[0] - p_before[0])**2 + (p_this[1] - p_before[1])**2).sqrt()
        d_next = ((p_this[0] - p_next[0])**2 + (p_this[1] - p_next[1])**2).sqrt()
        sr_term += (d_before - d_next)**2

    sr_term = sr_term.mean() / 24
    return sr_term


def get_d_list(pts):
    d_list_upper = []
    d_list_lower = []

    # from 17 to 11
    for i in range(6):
        j = 12 - 2*i
        p_this = (pts[:, j], pts[:, j + 1])
        p_next = (pts[:, j - 2], pts[:, j - 1])
        d_list_upper.append(((p_this[0] - p_next[0])**2 + (p_this[1] - p_next[1])**2).sqrt())

    # between 11 & 21
    p_this = (pts[:, 0], pts[:, 1])
    p_next = (pts[:, 16], pts[:, 17])
    d_list_upper.append(((p_this[0] - p_next[0])**2 + (p_this[1] - p_next[1])**2).sqrt())

    # from 21 to 27
    for i in range(6):
        j = 16 + 2*i
        p_this = (pts[:, j], pts[:, j + 1])
        p_next = (pts[:, j + 2], pts[:, j + 3])
        d_list_upper.append(((p_this[0] - p_next[0])**2 + (p_this[1] - p_next[1])**2).sqrt())

    # from 47 to 41
    for i in range(6):
        j = 60 - 2*i
        p_this = (pts[:, j], pts[:, j + 1])
        p_next = (pts[:, j - 2], pts[:, j - 1])
        d_list_lower.append(((p_this[0] - p_next[0])**2 + (p_this[1] - p_next[1])**2).sqrt())

    # between 41 & 31
    p_this = (pts[:, 48], pts[:, 49])
    p_next = (pts[:, 32], pts[:, 33])
    d_list_lower.append(((p_this[0] - p_next[0])**2 + (p_this[1] - p_next[1])**2).sqrt())

    # from 31 to 37
    for i in range(6):
        j = 32 + 2*i
        p_this = (pts[:, j], pts[:, j + 1])
        p_next = (pts[:, j + 2], pts[:, j + 3])
        d_list_lower.append(((p_this[0] - p_next[0])**2 + (p_this[1] - p_next[1])**2).sqrt())

    return d_list_upper, d_list_lower


def get_prime_list(input_list):
    prime_list = []
    L = len(input_list)
    for i in range(L-1):
        prime_list.append(input_list[i+1] - input_list[i])

    return prime_list


def get_SR_L2(pts):
    ### 1. Build d list ###
    d_list_upper, d_list_lower = get_d_list(pts)

    ### 2. Build d_prime list ###
    d_prime_list_upper = get_prime_list(d_list_upper)
    d_prime_list_lower = get_prime_list(d_list_lower)

    ### 3. Build d_two_prime list ###
    d_two_prime_list_upper = get_prime_list(d_prime_list_upper)
    d_two_prime_list_lower = get_prime_list(d_prime_list_lower)

    ### 4. Calc L2 Norm of d_two_prime_list ###
    sr_term = 0
    final_list = d_two_prime_list_upper + d_two_prime_list_lower
    for d in final_list:
        sr_term += d**2

    sr_term = sr_term.mean() / len(final_list)
    return sr_term


def get_SR_term(pts, mode='mse', target_pts=None, criterion=None):
    if mode == 'mse':
        return get_SR_MSE(pts)
    elif mode == 'l2':
        return get_SR_L2(pts)
    elif mode == 'sd_mse':
        out_d_list_upper, out_d_list_lower = get_d_list(pts)
        out_d_list = torch.stack(out_d_list_upper + out_d_list_lower, -1)
        target_d_list_upper, target_d_list_lower = get_d_list(target_pts)
        target_d_list = torch.stack(target_d_list_upper + target_d_list_lower, -1)
        return criterion(out_d_list, target_d_list)
    else:
        raise NotImplementedError


def _neg_loss(pred, gt):
    '''
        Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.ge(0.99).float()
    neg_inds = gt.lt(0.99).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def get_bbox_reg_loss(pred_bbox, gt, criterion):
    '''
        pred_bbox list of tensor(num_class x (batch x num_reg))
        gt tensor(batch x num_class x num_reg)
    '''
    reg_loss = 0
    _, num_class, _ = gt.size()

    for c in range(num_class):
        reg_loss += criterion(pred_bbox[c], gt[:, c])

    return reg_loss
