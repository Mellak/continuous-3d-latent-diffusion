from __future__ import annotations
import torch
import torch.nn.functional as F

def hinge_discriminator_loss(real_logits,fake_logits): return .5*(F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean())
def hinge_generator_loss(fake_logits): return -fake_logits.mean()
def adaptive_gan_weight(reconstruction_perceptual_loss,generator_gan_loss,last_layer_weight,eps=1e-4,maximum=1e4):
    rg=torch.autograd.grad(reconstruction_perceptual_loss,last_layer_weight,retain_graph=True,allow_unused=True)[0]
    gg=torch.autograd.grad(generator_gan_loss,last_layer_weight,retain_graph=True,allow_unused=True)[0]
    if rg is None or gg is None:return last_layer_weight.new_tensor(1.0)
    if torch.distributed.is_initialized():
        rg=rg.detach().clone();gg=gg.detach().clone()
        torch.distributed.all_reduce(rg);torch.distributed.all_reduce(gg)
        rg/=torch.distributed.get_world_size();gg/=torch.distributed.get_world_size()
    ratio=rg.norm()/(gg.norm()+float(eps)); ratio=torch.nan_to_num(ratio,nan=1.0,posinf=float(maximum),neginf=0.0)
    return ratio.clamp(0,float(maximum)).detach()
