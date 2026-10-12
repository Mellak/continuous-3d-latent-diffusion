from __future__ import annotations
import torch
import torch.nn.functional as F

class DiagonalGaussian3D:
    """Stable diagonal posterior matching the user's previous AE convention."""
    def __init__(self, mean:torch.Tensor, raw_variance:torch.Tensor, eps:float=1e-6):
        self.mean=mean
        self.variance=F.softplus(raw_variance.float())+float(eps)
        self.log_variance=torch.log(self.variance)
    def sample(self): return self.mean+torch.sqrt(self.variance)*torch.randn_like(self.mean)
    def rsample(self): return self.sample()
    def mode(self): return self.mean
    def kl(self):
        v=self.variance+self.mean.float().square()-1.0-self.log_variance
        return .5*v.sum(dim=1).mean()
