"""Small dependency-free 3-D SSIM loss; disabled by default in legacy configs."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def ssim3d(x,y,window=7,data_range=2.0,eps=1e-8):
    pad=window//2; mu_x=F.avg_pool3d(x,window,1,pad);mu_y=F.avg_pool3d(y,window,1,pad)
    vx=F.avg_pool3d(x*x,window,1,pad)-mu_x.square();vy=F.avg_pool3d(y*y,window,1,pad)-mu_y.square()
    vxy=F.avg_pool3d(x*y,window,1,pad)-mu_x*mu_y
    c1=(.01*data_range)**2;c2=(.03*data_range)**2
    val=((2*mu_x*mu_y+c1)*(2*vxy+c2))/((mu_x.square()+mu_y.square()+c1)*(vx+vy+c2)+eps)
    return val.mean()
def ssim3d_loss(x,y,**kw):return 1-ssim3d(x,y,**kw)
