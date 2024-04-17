import torch
import numpy as np
from copy import deepcopy
def mixup_soft(data, target=None, max=None, alpha=1.0):
    # target必须是one-hot的

    # Generate random indices to shuffle the images
    indices = torch.randperm(len(data)).to(data.device)
    shuffled_data = data[indices]

    # Generate image weight (minimum 0.4 and maximum 0.6)
    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    
    # Weighted Mixup
    mixedup_data = lam*data + (1 - lam)*shuffled_data

    shuffled_target = target[indices]
    mixedup_target = lam*target + (1 - lam)*shuffled_target

    if max is not None:
        return mixedup_data, mixedup_target, max, max[indices]
    
    else:
        return mixedup_data, mixedup_target

def mixup_hard(data, indices, alpha=1.0):
    # Generate random indices to shuffle the images
    #indices = torch.randperm(len(data)).to(data.device)
    shuffled_data = data[indices]

    # Generate image weight (minimum 0.4 and maximum 0.6)
    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    
    # Weighted Mixup
    mixedup_data = lam*data + (1 - lam)*shuffled_data
    
    return mixedup_data, lam

'''   
def mixup_hard(data, target, max=None, alpha=1.0):
    # Generate random indices to shuffle the images
    indices = torch.randperm(len(data)).to(data.device)
    shuffled_data = data[indices]

    # Generate image weight (minimum 0.4 and maximum 0.6)
    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    
    # Weighted Mixup
    mixedup_data = lam*data + (1 - lam)*shuffled_data
    if max is not None:
        return mixedup_data, target, target[indices], max, max[indices], lam
    else:
        return mixedup_data, target, target[indices], lam
'''

def cutmix_soft(data, target, max=None, alpha=1.0):
	
    #CutMix augmentation implementation.
    #alpha: hyperparameter controlling the strength of CutMix regularization
    #lam: Mixing ratio of types A and B

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    target = lam * target + (1-lam) * target[indices]

    if max is not None:
        return data, target, max, max[indices]
    else:
        return data, target

def cutmix_hard(data, indices, bbox, alpha=1.0):
	
    data_cut = deepcopy(data)
    #CutMix augmentation implementation.
    #alpha: hyperparameter controlling the strength of CutMix regularization
    #lam: Mixing ratio of types A and B

    #indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]

    idx = range(bbox.shape[0])
    data_cut = data * (1-bbox.unsqueeze(1)) + shuffled_data * bbox.unsqueeze(1)

    # Adjust lambda to exactly match pixel ratio

    return data_cut

def rand_bbox(size, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    #import ipdb
    #ipdb.set_trace()
    lam = np.max([lam, 1 - lam])
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    #cut_w = np.int(W * cut_rat)
    #cut_h = np.int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx1- bbx2) * (bby1- bby2) / (W * H))

    mask = torch.zeros(W, H)
    mask[bbx1:bbx2, bby1:bby2] = 1

    return mask, lam
'''
def cutmix_hard(data, target, max=None, alpha=1.0):
	
    #CutMix augmentation implementation.
    #alpha: hyperparameter controlling the strength of CutMix regularization
    #lam: Mixing ratio of types A and B

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]

    lam = np.random.beta(alpha, alpha)
    #import ipdb
    #ipdb.set_trace()
    lam = np.max([lam, 1 - lam])

    import ipdb
    ipdb.set_trace()

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    if max is not None:
        return data, target, target[indices], max, max[indices], lam
    else:
        return data, target, target[indices], lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
'''
