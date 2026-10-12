from __future__ import annotations
import torch

def stable_covariance_loss(latent_mode:torch.Tensor,eps:float=1e-3,normalized_value_clip:float=10.0):
    if latent_mode.ndim<3: raise ValueError('Expected [B,C,*spatial].')
    c=latent_mode.shape[1]; x=latent_mode.float().reshape(latent_mode.shape[0],c,-1).permute(1,0,2).reshape(c,-1)
    x=torch.nan_to_num(x,nan=0.0,posinf=1e4,neginf=-1e4); x=x-x.mean(1,keepdim=True)
    x=x*torch.rsqrt(x.var(1,keepdim=True,unbiased=False)+float(eps))
    if normalized_value_clip>0:x=x.clamp(-normalized_value_clip,normalized_value_clip)
    corr=x@x.t()/max(1,x.shape[1]); eye=torch.eye(c,device=x.device,dtype=x.dtype)
    return torch.nan_to_num(((corr*(1-eye)).square()).mean(),nan=0.0,posinf=1e4,neginf=1e4)
