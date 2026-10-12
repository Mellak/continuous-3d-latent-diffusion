import torch.nn.functional as F

def l1_loss(pred,target): return F.l1_loss(pred,target)
def l2_loss(pred,target): return F.mse_loss(pred,target)
