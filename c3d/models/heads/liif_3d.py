"""LIIF rendering head of CoRe-AE: 3-D adaptation of the LIIF local-ensemble decoder (Eq. 10)."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import sample_fp32


class MLP(nn.Module):
    def __init__(self,in_dim:int,out_dim:int=1,hidden_dims:Sequence[int]=(128,128,128),act:str='relu',dropout:float=0.0):
        super().__init__(); acts={'relu':nn.ReLU,'gelu':nn.GELU,'silu':nn.SiLU}
        if act not in acts: raise ValueError(f'Unknown activation {act}')
        layers:List[nn.Module]=[]; prev=in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,int(h)),acts[act]()]
            if dropout>0: layers.append(nn.Dropout(float(dropout)))
            prev=int(h)
        layers.append(nn.Linear(prev,int(out_dim))); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


def coord_grid_3d(spatial:Sequence[int],device,dtype):
    d,h,w=(int(v) for v in spatial)
    z=(torch.arange(d,device=device,dtype=torch.float32)+.5)*(2/d)-1
    y=(torch.arange(h,device=device,dtype=torch.float32)+.5)*(2/h)-1
    x=(torch.arange(w,device=device,dtype=torch.float32)+.5)*(2/w)-1
    zz,yy,xx=torch.meshgrid(z,y,x,indexing='ij')
    return torch.stack((xx,yy,zz),-1).reshape(-1,3)


def cell_3d(spatial:Sequence[int],device,dtype):
    d,h,w=(int(v) for v in spatial)
    return torch.tensor([2/w,2/h,2/d],device=device,dtype=torch.float32).view(1,1,3)

def coord_chunk_3d(spatial:Sequence[int],start:int,end:int,device,dtype):
    """Create only [start:end] pixel-center coordinates in flattened DHW order."""
    d,h,w=(int(v) for v in spatial)
    idx=torch.arange(int(start),int(end),device=device,dtype=torch.long)
    z=torch.div(idx,h*w,rounding_mode='floor')
    rem=idx-z*(h*w)
    y=torch.div(rem,w,rounding_mode='floor')
    x=rem-y*w
    return torch.stack(((x.float()+.5)*(2.0/w)-1.0,(y.float()+.5)*(2.0/h)-1.0,(z.float()+.5)*(2.0/d)-1.0),dim=-1)


def feature_coord_grid(feat:torch.Tensor):
    b,_,d,h,w=feat.shape
    return coord_grid_3d((d,h,w),feat.device,torch.float32).view(d,h,w,3).permute(3,0,1,2).unsqueeze(0).expand(b,-1,-1,-1,-1)


@dataclass
class LIIFHeadConfig:
    out_channels:int=1
    local_ensemble:bool=True
    feat_unfold:bool=False
    use_cell:bool=True
    padding_mode:str='border'
    mlp_hidden:Tuple[int,...]=(128,128,128)
    mlp_act:str='relu'
    mlp_dropout:float=0.0
    render_chunk:int=300_000


class LIIFHead3D(nn.Module):
    def __init__(self,in_channels:int,cfg:LIIFHeadConfig):
        super().__init__(); self.cfg=cfg; self.in_channels=int(in_channels)
        feat_dim=self.in_channels*(27 if cfg.feat_unfold else 1)
        self.imnet=MLP(feat_dim+3+(3 if cfg.use_cell else 0),cfg.out_channels,cfg.mlp_hidden,cfg.mlp_act,cfg.mlp_dropout)
    def get_last_layer_weight(self): return self.imnet.net[-1].weight
    def _unfold(self,feat):
        if not self.cfg.feat_unfold:return feat
        p=F.pad(feat,(1,1,1,1,1,1),mode='replicate'); _,_,d,h,w=feat.shape; xs=[]
        for dz in range(3):
            for dy in range(3):
                for dx in range(3): xs.append(p[:,:,dz:dz+d,dy:dy+h,dx:dx+w])
        return torch.cat(xs,1)
    @staticmethod
    def _sample(field,coord,mode='nearest',padding='border'):
        b=field.shape[0]; grid=coord.view(b,-1,1,1,3)
        out=sample_fp32(field,grid,mode=mode,padding_mode=padding,align_corners=False)
        return out.flatten(2).transpose(1,2).contiguous()
    def _query_prepared(self,feat_u,fcoord,coord,cell:Optional[torch.Tensor]=None):
        if feat_u.ndim!=5 or coord.ndim!=3 or coord.shape[-1]!=3: raise ValueError('Expected feat [B,C,D,H,W], coord [B,N,3].')
        if self.cfg.use_cell and cell is None: raise ValueError('use_cell=True but cell is None')
        coord=coord.float(); cell=None if cell is None else cell.float()
        b,_,d,h,w=feat_u.shape
        shifts=list(product((-1,1),repeat=3)) if self.cfg.local_ensemble else [(0,0,0)]
        radii=(1/w,1/h,1/d); scale=coord.new_tensor([w,h,d]).view(1,1,3); eps=1e-6
        preds={}; areas={}
        for sx,sy,sz in shifts:
            shifted=coord.clone(); shifted[...,0]+=sx*radii[0]+eps; shifted[...,1]+=sy*radii[1]+eps; shifted[...,2]+=sz*radii[2]+eps
            shifted.clamp_(-1+1e-6,1-1e-6)
            qfeat=self._sample(feat_u,shifted,'nearest',self.cfg.padding_mode)
            qcoord=self._sample(fcoord,shifted,'nearest',self.cfg.padding_mode)
            rel=(coord-qcoord)*scale
            parts=[qfeat,rel]
            if self.cfg.use_cell: parts.append(cell*scale)
            pred=self.imnet(torch.cat(parts,-1)); key=(sx,sy,sz); preds[key]=pred
            areas[key]=(rel[...,0].abs()*rel[...,1].abs()*rel[...,2].abs()+1e-9)
        if not self.cfg.local_ensemble:return next(iter(preds.values()))
        total=torch.stack(list(areas.values())).sum(0); out=0
        for key,pred in preds.items():
            opposite=tuple(-v for v in key); out=out+pred*(areas[opposite]/total).unsqueeze(-1)
        return out
    def query(self,feat,coord,cell:Optional[torch.Tensor]=None):
        feat_u=self._unfold(feat); fcoord=feature_coord_grid(feat_u)
        return self._query_prepared(feat_u,fcoord,coord,cell)
    def render(self,feat,spatial:Sequence[int],chunk:Optional[int]=None):
        spatial=tuple(int(v) for v in spatial); b=feat.shape[0]; ch=int(chunk or self.cfg.render_chunk); total=int(spatial[0]*spatial[1]*spatial[2])
        feat_u=self._unfold(feat); fcoord=feature_coord_grid(feat_u); cell0=cell_3d(spatial,feat.device,torch.float32) if self.cfg.use_cell else None
        outs=[]
        for s in range(0,total,ch):
            e=min(total,s+ch); coord=coord_chunk_3d(spatial,s,e,feat.device,torch.float32).unsqueeze(0).expand(b,-1,3)
            cell=None if cell0 is None else cell0.expand(b,e-s,3)
            outs.append(self._query_prepared(feat_u,fcoord,coord,cell))
        d,h,w=spatial; return torch.cat(outs,1).transpose(1,2).contiguous().view(b,self.cfg.out_channels,d,h,w)
