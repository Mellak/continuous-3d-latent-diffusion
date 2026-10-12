from __future__ import annotations
import torch
@torch.no_grad()
def sample(model,shape,flow,steps=50,method='heun',seed=314159,device='cuda',stop=None):
    device=torch.device(device);gen=torch.Generator(device=device).manual_seed(seed);z=torch.randn(shape,generator=gen,device=device,dtype=torch.float32)*float(flow.get('noise_scale',1.0));grid=torch.linspace(0,1,int(steps)+1,device=device,dtype=torch.float32)
    for i in range(int(steps)):
        if stop and stop():return None
        t=grid[i].expand(shape[0]);pred=model(z,t).float()
        if i==int(steps)-1:return pred
        dt=grid[i+1]-grid[i];v=(pred-z)/(1-grid[i]).clamp_min(float(flow.get('t_eps',.05)));proposal=z+dt*v
        if method=='heun':
            next_pred=model(proposal,grid[i+1].expand(shape[0])).float();next_v=(next_pred-proposal)/(1-grid[i+1]).clamp_min(float(flow.get('t_eps',.05)));z=z+dt*(v+next_v)*.5
        elif method=='euler':z=proposal
        else:raise ValueError(method)
    raise ValueError('sampling steps must be positive')
