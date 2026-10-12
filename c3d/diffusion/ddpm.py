from __future__ import annotations
from dataclasses import dataclass
import numpy as np,torch
@dataclass
class DDPMSchedule:
    num_steps:int;betas:torch.Tensor;alphas:torch.Tensor;alphas_cumprod:torch.Tensor;alphas_cumprod_prev:torch.Tensor;sqrt_alphas_cumprod:torch.Tensor;sqrt_one_minus_alphas_cumprod:torch.Tensor;posterior_variance:torch.Tensor;posterior_mean_coef1:torch.Tensor;posterior_mean_coef2:torch.Tensor
def make_schedule(num_steps=1000,beta_start=1e-4,beta_end=2e-2,schedule_type='linear'):
    if schedule_type=='linear':betas=torch.linspace(beta_start,beta_end,num_steps,dtype=torch.float32).clamp_max(.999)
    elif schedule_type=='cosine':
        x=np.linspace(0,num_steps,num_steps+1);ab=np.cos(((x/num_steps)+.008)/1.008*np.pi/2)**2;ab/=ab[0];betas=torch.tensor(np.clip(1-ab[1:]/ab[:-1],1e-8,.999),dtype=torch.float32)
    else:raise ValueError(schedule_type)
    a=1-betas;ab=torch.cumprod(a,0);prev=torch.cat((torch.ones(1),ab[:-1]));pv=betas*(1-prev)/(1-ab)
    return DDPMSchedule(num_steps,betas,a,ab,prev,ab.sqrt(),(1-ab).sqrt(),pv,betas*prev.sqrt()/(1-ab),(1-prev)*a.sqrt()/(1-ab))
def extract(a,t,shape):
    out=a.to(t.device)[t]
    while out.ndim<len(shape):out=out.unsqueeze(-1)
    return out
def objective(model,x0,schedule,prediction_type='eps'):
    b=x0.shape[0];t=torch.randint(0,schedule.num_steps,(b,),device=x0.device);noise=torch.randn_like(x0);xt=extract(schedule.sqrt_alphas_cumprod,t,x0.shape)*x0+extract(schedule.sqrt_one_minus_alphas_cumprod,t,x0.shape)*noise
    pred=model(xt,t);target=noise if prediction_type=='eps' else x0
    return torch.nn.functional.mse_loss(pred,target),{'t_mean':t.float().mean().detach()}
@torch.no_grad()
def sample(model,shape,schedule,prediction_type='eps',seed=314159,device='cuda',stop=None):
    device=torch.device(device);g=torch.Generator(device=device).manual_seed(seed);x=torch.randn(shape,generator=g,device=device)
    for ti in reversed(range(schedule.num_steps)):
        if stop and stop():return None
        t=torch.full((shape[0],),ti,device=device,dtype=torch.long);out=model(x,t)
        if prediction_type=='eps':x0=(x-extract(schedule.sqrt_one_minus_alphas_cumprod,t,x.shape)*out)/extract(schedule.sqrt_alphas_cumprod,t,x.shape).clamp_min(1e-8)
        else:x0=out
        mean=extract(schedule.posterior_mean_coef1,t,x.shape)*x0+extract(schedule.posterior_mean_coef2,t,x.shape)*x
        if ti>0:x=mean+extract(schedule.posterior_variance,t,x.shape).clamp_min(1e-20).sqrt()*torch.randn(x.shape,generator=g,device=device)
        else:x=mean
    return x
