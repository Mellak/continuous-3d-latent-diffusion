"""CoRe-AE: 3-D KL autoencoder with a continuous rendering head (LIIF or CLIF)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import torch
from torch import nn
from c3d.models.checkpointing import run_checkpoint

from .encoder_3d import StridedEncND
from .feature_decoder_3d import DecCNN, DecCNNConfig, StageCfg
from .gaussian import DiagonalGaussian3D
from c3d.models.heads.liif_3d import LIIFHead3D, LIIFHeadConfig
from c3d.models.heads.clif_3d import CLIFHead3D, CLIFHeadConfig

# Earlier checkpoints and configs call the LIIF head 'liif_corr'.
HEAD_ALIASES={'liif_corr':'liif'}
def canonical_head(name): return HEAD_ALIASES.get(str(name),str(name))


def _tuple(v): return tuple(v) if isinstance(v,(list,tuple)) else tuple(int(x) for x in str(v).replace('x',',').split(','))

class CoReAE(nn.Module):
    def __init__(self,cfg:Dict[str,Any]):
        super().__init__(); self.cfg=cfg
        enc=cfg['encoder']; dec=cfg['feature_decoder']; head=cfg['head']; self.head_name=canonical_head(head['name'])
        self.encoder=StridedEncND(dim=3,in_channels=int(enc.get('in_channels',1)),base_ch=int(enc['base_channels']),latent_ch=int(enc['latent_channels']),norm=enc.get('norm','gn'),gn_groups=int(enc.get('gn_groups',8)))
        stages=[StageCfg(int(c),int(b),int(s)) for c,b,s in zip(dec['stage_channels'],dec['stage_blocks'],dec['stage_strides'])]
        self.feature_decoder=DecCNN(DecCNNConfig(dim=3,in_channels=int(enc['latent_channels']),stages=stages,
            out_channels=None if int(dec.get('out_channels',-1))<0 else int(dec['out_channels']),block=dec.get('block','basic'),norm=dec.get('norm','gn'),
            gn_groups=int(dec.get('gn_groups',8)),act=dec.get('act','silu'),dropout=float(dec.get('dropout',0.0)),drop_path_rate=float(dec.get('drop_path_rate',0.0)),
            in_proj=bool(dec.get('in_proj',True)),out_proj=bool(dec.get('out_proj',False)),residual_scale=float(dec.get('residual_scale',1.0))))
        c=self.feature_decoder.out_channels; hc=head.get('config',{})
        if self.head_name=='liif': self.head=LIIFHead3D(c,LIIFHeadConfig(**hc))
        elif self.head_name=='clif':
            # YAML uses user-facing residual_blocks/render_tile names.
            mapped=dict(hc)
            if 'residual_blocks' in mapped: mapped['num_res_blocks']=mapped.pop('residual_blocks')
            if 'convolution_padding_mode' in mapped: mapped['conv_padding_mode']=mapped.pop('convolution_padding_mode')
            if 'render_tile' in mapped: mapped['tile_size']=tuple(mapped.pop('render_tile'))
            if 'training_patch' in mapped: mapped.pop('training_patch')
            mapped.pop('scale_min',None); mapped.pop('scale_max',None); mapped.pop('render_chunk',None)
            mapped['dim']=3
            self.head=CLIFHead3D(c,CLIFHeadConfig(**mapped))
        else: raise ValueError(f'Unknown head: {self.head_name}')

    @property
    def latent_channels(self): return int(self.cfg['encoder']['latent_channels'])
    @property
    def feature_channels(self): return int(self.feature_decoder.out_channels)

    def posterior(self,x):
        mean,raw=self.encoder(x); return DiagonalGaussian3D(mean,raw)
    def encode(self,x,sample:bool=True):
        posterior=self.posterior(x); return (posterior.rsample() if sample else posterior.mode()),posterior
    def decode_features(self,z): return self.feature_decoder(z)
    def encode_features(self,x,sample_latent:bool=True):
        z,p=self.encode(x,sample_latent); return self.decode_features(z),z,p
    def query(self,features,coordinates,cells=None):
        if self.head_name=='clif': return self.head(features,coordinates,cells)
        return self.head.query(features,coordinates,cells)
    def _forward_training_outputs(self,x,coordinates,cells,sample_latent,
                                  checkpoint_blocks=False,query_chunk=16384):
        # Return only reconstruction and posterior parameters. Returning the
        # sampled latent/coarse features would retain unnecessary outputs of
        # the checkpointed region. All losses are evaluated by the trainer.
        mean,raw=self.encoder(x,checkpoint_blocks=checkpoint_blocks)
        # Share the FP32 variance input between sampling and the external KL
        # loss, so their gradients merge before casting back to the encoder's
        # autocast dtype (the original posterior also transforms raw in FP32).
        raw=raw.float()
        posterior=DiagonalGaussian3D(mean,raw)
        z=posterior.rsample() if sample_latent else posterior.mode()
        features=self.feature_decoder(z,checkpoint_blocks=checkpoint_blocks)
        if self.head_name=='clif':
            pred=self.head(features,coordinates,cells,checkpoint_blocks=checkpoint_blocks)
        else:
            # All queried points contribute to ONE mean L1 and ONE optimizer
            # step. Chunking bounds saved/recomputed head activations only.
            if query_chunk<1:raise ValueError('training_query_chunk must be positive')
            parts=[]
            for start in range(0,coordinates.shape[1],query_chunk):
                co=coordinates[:,start:start+query_chunk]
                ce=None if cells is None else cells[:,start:start+query_chunk]
                parts.append(run_checkpoint(self.head.query,features,co,ce,enabled=checkpoint_blocks))
            pred=torch.cat(parts,dim=1)
        return pred,mean,raw

    def forward(self,x,coordinates,cells=None,*,sample_latent=True,
                training_outputs=False,activation_checkpointing=False,training_query_chunk=16384):
        """Keep the original four-output API unless training_outputs is set.

        The training API returns (prediction, posterior_mean, raw_variance).
        Checkpoint small blocks/query chunks, never the entire AE or DDP.
        Non-reentrant checkpointing supports adaptive GAN autograd.grad calls
        and inputs which do not require gradients. Preserve RNG for posterior
        sampling/dropout, and let PyTorch restore the forward autocast context.
        """
        if activation_checkpointing and not training_outputs:
            raise ValueError('Activation checkpointing requires training_outputs=True')
        if training_outputs:
            enabled=bool(activation_checkpointing and self.training and torch.is_grad_enabled())
            return self._forward_training_outputs(x,coordinates,cells,sample_latent,
                checkpoint_blocks=enabled,query_chunk=int(training_query_chunk))
        f,z,p=self.encode_features(x,sample_latent); return self.query(f,coordinates,cells),z,p,f
    def decode_latent(self,z,spatial:Sequence[int],**kwargs):
        f=self.decode_features(z); return self.render_from_features(f,spatial,**kwargs)
    def render_from_features(self,features,spatial:Sequence[int],**kwargs):
        if self.head_name=='clif': return self.head.render(features,tuple(int(v) for v in spatial),tile_size=kwargs.get('tile_size'),halo=kwargs.get('halo'))
        return self.head.render(features,tuple(int(v) for v in spatial),chunk=kwargs.get('chunk'))
    def get_last_layer_weight(self):
        if hasattr(self.head,'get_last_layer_weight'): return self.head.get_last_layer_weight()
        raise AttributeError('Head has no last-layer accessor')


def build_autoencoder(cfg:Dict[str,Any]):
    return CoReAE(cfg['model'] if 'model' in cfg else cfg)


def _strip_prefix(state:Dict[str,torch.Tensor],prefix:str):
    return {k[len(prefix):] if k.startswith(prefix) else k:v for k,v in state.items()}


def _is_safetensors(path):
    with open(path,'rb') as f: head=f.read(9)
    return len(head)==9 and head[8:9]==b'{'


def load_checkpoint_state(path:str|Path,map_location='cpu'):
    if _is_safetensors(path):
        from safetensors.torch import load_file
        state=load_file(str(path)); return state,{'model':state}
    obj=torch.load(str(path),map_location=map_location,weights_only=False)
    if isinstance(obj,dict):
        for key in ('model','state_dict','generator'):
            if key in obj and isinstance(obj[key],dict): return obj[key],obj
    if isinstance(obj,dict): return obj,obj
    raise ValueError(f'Unsupported checkpoint format: {path}')


def remap_legacy_state(state:Dict[str,torch.Tensor],target_head:str,include_head:bool=True):
    """Map old VAE_LIIFND/adaptive checkpoint keys to this project.

    Reusable mappings:
      encoder.* -> encoder.*
      dec_cnn.* -> feature_decoder.*
      backbone.encoder.* -> encoder.*
      backbone.feature_decoder.* -> feature_decoder.*
    """
    out={}
    for k,v in state.items():
        if k.startswith('module.'): k=k[7:]
        if k.startswith('backbone.encoder.'): nk='encoder.'+k[len('backbone.encoder.'):]
        elif k.startswith('backbone.feature_decoder.'): nk='feature_decoder.'+k[len('backbone.feature_decoder.'):]
        elif k.startswith('encoder.'): nk=k
        elif k.startswith('dec_cnn.'): nk='feature_decoder.'+k[len('dec_cnn.'):]
        elif k.startswith('feature_decoder.'): nk=k
        else: continue
        out[nk]=v
    return out


def initialize_from_checkpoint(model:CoReAE,path:str|Path,*,include_head:bool=False,strict:bool=False):
    state,meta=load_checkpoint_state(path)
    mapped=remap_legacy_state(state,model.head_name,include_head=include_head)
    target=model.state_dict()
    required={k for k in target if k.startswith(('encoder.', 'feature_decoder.')) or include_head}
    missing=sorted(required-set(mapped))
    mismatched=[k for k in required & set(mapped) if tuple(target[k].shape)!=tuple(mapped[k].shape)]
    if missing or mismatched:
        raise ValueError(f"Checkpoint cannot initialize required components: missing={missing[:12]}, shape_mismatch={mismatched[:12]}")
    result=model.load_state_dict({k:mapped[k] for k in required},strict=include_head)
    return {'loaded_keys':len(required),'missing_keys':list(result.missing_keys),'source':str(path)}


def _canonical_cfg(model_cfg):
    model_cfg=dict(model_cfg); head=dict(model_cfg.get('head',{}))
    if 'name' in head: head['name']=canonical_head(head['name'])
    model_cfg['head']=head; return model_cfg


def load_complete_model(cfg, path, device='cpu'):
    model=build_autoencoder(cfg).to(device)
    state,meta=load_checkpoint_state(path, map_location='cpu')
    if isinstance(meta,dict) and 'config' in meta:
        saved=meta['config'].get('model',meta['config'])
        requested=cfg.get('model',cfg)
        if _canonical_cfg(saved) != _canonical_cfg(requested):
            raise ValueError("AE model config differs from checkpoint config; pass the checkpoint's saved config.")
    state=_strip_prefix(state,'module.')
    if set(state) != set(model.state_dict()):
        state=remap_legacy_state(state,model.head_name,include_head=True)
    model.load_state_dict(state,strict=True)
    return model.eval()


def set_training_mode(model:CoReAE,mode:str):
    mode=str(mode).lower()
    if mode not in ('scratch','decoder','finetune'): raise ValueError('mode must be scratch, decoder, or finetune')
    for p in model.parameters(): p.requires_grad_(True)
    if mode=='decoder':
        for p in model.encoder.parameters(): p.requires_grad_(False)
    return model
