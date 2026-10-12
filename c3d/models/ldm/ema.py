from __future__ import annotations
from contextlib import contextmanager
import torch
class EMA:
    def __init__(self,model,decay=.9999):self.decay=float(decay);self.shadow={k:v.detach().clone() for k,v in model.state_dict().items() if torch.is_floating_point(v)}
    @torch.no_grad()
    def update(self,model):
        for k,v in model.state_dict().items():
            if k in self.shadow:self.shadow[k].lerp_(v.detach(),1-self.decay)
    def state_dict(self):return {'decay':self.decay,'shadow':self.shadow}
    def load_state_dict(self,s):self.decay=float(s['decay']);self.shadow={k:v.to(self.shadow[k].device) for k,v in s['shadow'].items()}
    @contextmanager
    def applied(self,model):
        state=model.state_dict();backup={k:state[k].detach().clone() for k in self.shadow};model.load_state_dict({**state,**self.shadow},strict=False)
        try:yield
        finally:model.load_state_dict({**model.state_dict(),**backup},strict=False)
