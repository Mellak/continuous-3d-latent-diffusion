from __future__ import annotations
import inspect
import torch
from torch import nn
from .unet_flash_3d import UNetModel

class JiTPreconditionedModel(nn.Module):
    """Latent analogue of the supplied Haar FlowWPC wrapper."""
    def __init__(self,backbone,channel_mean,channel_variance,noise_scale=1.0,time_embedding_scale=1.0,variance_floor=1e-12):
        super().__init__();self.backbone=backbone;self.noise_scale=float(noise_scale);self.time_scale=float(time_embedding_scale);self.floor=float(variance_floor)
        c=len(channel_mean);self.register_buffer('mean',torch.tensor(channel_mean,dtype=torch.float32).view(1,c,1,1,1));self.register_buffer('variance',torch.tensor(channel_variance,dtype=torch.float32).view(1,c,1,1,1))
    def normalize(self,z,t):
        with torch.autocast(z.device.type,enabled=False):
            tb=t.float().view(-1,1,1,1,1);var=tb.square()*self.variance+(1-tb).square()*self.noise_scale**2
            return (z.float()-tb*self.mean)*var.clamp_min(self.floor).rsqrt()
    def forward(self,z,t):return self.backbone(self.normalize(z,t),t.float()*self.time_scale)

def build_ldm(model_cfg,objective_cfg,stats):
    cfg = dict(model_cfg)
    architecture = cfg.pop('architecture', 'unet_flash_3d')
    if architecture != 'unet_flash_3d':
        raise ValueError('This training path uses unet_flash_3d.UNetModel; start a fresh flash-U-Net run')
    # UNetModel accepts **kwargs. Reject stale/unsupported fields here so a
    # boundary configuration cannot silently appear to configure this model.
    allowed = set(inspect.signature(UNetModel.__init__).parameters) - {'self', 'kwargs', 'config'}
    unsupported = set(cfg) - allowed
    if unsupported:
        raise ValueError('Unsupported unet_flash_3d settings: ' + ', '.join(sorted(unsupported))
                         + '. Remove boundary-only fields and use a fresh diffusion run.')
    if cfg.get('use_checkpoint', False):
        raise ValueError('The supplied UNet forward uses ordinary autograd; set use_checkpoint=false')
    if cfg.get('use_fp16', False):
        raise ValueError('Keep use_fp16=false and select BF16/FP16 autocast under training.precision')
    shape = tuple(int(v) for v in stats['latent_shape'])
    if len(shape) != 4 or cfg.get('dims', 3) != 3 or not cfg.get('resample_3d', True):
        raise ValueError('Expected C,D,H,W latents with dims=3 and resample_3d=true')
    factor = 2 ** (len(cfg.get('channel_mult', (1, 2, 2, 2, 4))) - 1)
    if any(n < factor or n % factor for n in shape[1:]):
        raise ValueError(f'Each latent spatial dimension must be divisible by {factor}: {shape}')
    for key in ('in_channels', 'out_channels'):
        if key in cfg and cfg[key] != shape[0]:
            raise ValueError(f'{key} must match latent channels ({shape[0]})')
        cfg[key] = shape[0]
    cfg.setdefault('image_size', shape[-1])
    backbone=UNetModel(**cfg)
    if objective_cfg['name']=='jit_x0v':
        return JiTPreconditionedModel(backbone,stats['per_channel_mean_normalized'],stats['per_channel_variance_normalized'],objective_cfg.get('noise_scale',1.0),objective_cfg.get('time_embedding_scale',1.0),objective_cfg.get('variance_floor',1e-12))
    return backbone
