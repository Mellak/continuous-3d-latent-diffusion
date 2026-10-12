from __future__ import annotations
from typing import Optional
import torch
from torch import nn

class MONAIPerceptualLoss3D(nn.Module):
    """Same MONAI Generative fake-3D perceptual setup as the previous CLIF trainer."""
    def __init__(self,network_type='squeeze',fake_3d_ratio=.5,cache_dir:Optional[str]=None,pretrained=True):
        super().__init__(); P=None; errors=[]
        try:
            from generative.losses import PerceptualLoss as P
        except Exception as e: errors.append(f'generative.losses: {e!r}')
        if P is None:
            try: from monai.generative.losses import PerceptualLoss as P
            except Exception as e: errors.append(f'monai.generative.losses: {e!r}')
        if P is None: raise ImportError('MONAI Generative PerceptualLoss unavailable. '+' | '.join(errors))
        kw=dict(spatial_dims=3,network_type=str(network_type),is_fake_3d=True,fake_3d_ratio=float(fake_3d_ratio),pretrained=bool(pretrained))
        if cache_dir:kw['cache_dir']=str(cache_dir)
        try:self.metric=P(**kw)
        except TypeError:
            kw.pop('cache_dir',None);kw.pop('pretrained',None);self.metric=P(**kw)
        self.metric.eval()
        for p in self.metric.parameters():p.requires_grad_(False)
    def train(self,mode=True):super().train(False);self.metric.eval();return self
    def forward(self,pred,target):
        pred=pred.float();target=target.float()
        if pred.shape[1]==1:pred=pred.repeat(1,3,1,1,1);target=target.repeat(1,3,1,1,1)
        return self.metric(pred,target).mean()
