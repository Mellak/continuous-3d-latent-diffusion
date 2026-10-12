from .heun import sample as _sample
def sample(model,shape,flow,steps=50,seed=314159,device='cuda',stop=None):return _sample(model,shape,flow,steps,'euler',seed,device,stop)
