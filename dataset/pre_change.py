import torch

def get_mean_and_std(pre_predict, index):
    pre = pre_predict[index]
    pre_mean = torch.mean(pre, dim=1, keepdim=True)
    pre_std = torch.std(pre, dim=1, keepdim=True)

    return pre_mean,pre_std

def predict_update(pre_predict, max_probs, index):
    pre = pre_predict[index]
    #ipdb.set_trace()
    pre = torch.cat([pre[:,1:], max_probs.unsqueeze(1)], dim=1)
    pre_predict[index,:] = pre
    return pre_predict

def get_mean_and_std_all(pre_predict):
    pre_mean = torch.mean(pre_predict)
    pre_std = torch.std(pre_predict)

    return pre_mean,pre_std