from __future__ import annotations
import random
from typing import Sequence
import torch
import torch.nn.functional as F

def choose_resolution(native:Sequence[int], choices):
    if not choices:return tuple(int(v) for v in native)
    item=random.choice(choices)
    if isinstance(item,int):return (item,item,item)
    return tuple(int(v) for v in item)

def sample_liif_coordinates(batch,count,device,dtype,jitter=0.0):
    coord=torch.rand(batch,int(count),3,device=device,dtype=torch.float32)*2-1
    if jitter>0:coord=(coord+(torch.rand_like(coord)-.5)*float(jitter)).clamp(-1+1e-6,1-1e-6)
    return coord

def cells_for_resolution(spatial,batch,count,device,dtype):
    d,h,w=(int(v) for v in spatial); c=torch.tensor([2/w,2/h,2/d],device=device,dtype=torch.float32).view(1,1,3)
    return c.expand(batch,int(count),3)

def sample_volume_points(volume,coord,align_corners=False,padding_mode='border'):
    b=volume.shape[0]; grid=coord.view(b,-1,1,1,3)
    return F.grid_sample(volume,grid,mode='bilinear',padding_mode=padding_mode,align_corners=align_corners).flatten(2).transpose(1,2).contiguous()

def random_dense_patch_coordinates(batch,full_spatial,interior_patch,halo,scale_min,scale_max,device,dtype):
    """Dense CLIF crop at a random effective scale. Returns coord/cell with halo."""
    dtype=torch.float32
    d,h,w=(int(v) for v in full_spatial); pd,ph,pw=(int(v) for v in interior_patch); halo=int(halo)
    od,oh,ow=pd+2*halo,ph+2*halo,pw+2*halo
    coords=[];cells=[];scales=[]
    for _ in range(batch):
        scale=random.uniform(float(scale_min),float(scale_max));scales.append(scale)
        # effective output lattice cell; scale>1 means finer requested lattice
        eff=(max(1,int(round(d*scale))),max(1,int(round(h*scale))),max(1,int(round(w*scale))))
        # sample center in normalized continuous space while keeping patch inside [-1,1]
        cd=2/eff[0];ch=2/eff[1];cw=2/eff[2]
        half=((od-1)*cd/2,(oh-1)*ch/2,(ow-1)*cw/2)
        cz=random.uniform(-1+half[0],1-half[0]) if half[0]<1 else 0
        cy=random.uniform(-1+half[1],1-half[1]) if half[1]<1 else 0
        cx=random.uniform(-1+half[2],1-half[2]) if half[2]<1 else 0
        z=(torch.arange(od,device=device,dtype=dtype)-(od-1)/2)*cd+cz
        y=(torch.arange(oh,device=device,dtype=dtype)-(oh-1)/2)*ch+cy
        x=(torch.arange(ow,device=device,dtype=dtype)-(ow-1)/2)*cw+cx
        zz,yy,xx=torch.meshgrid(z,y,x,indexing='ij');co=torch.stack((xx,yy,zz),-1).clamp(-1+1e-6,1-1e-6)
        ce=torch.empty_like(co);ce[...,0]=cw;ce[...,1]=ch;ce[...,2]=cd;coords.append(co);cells.append(ce)
    return torch.stack(coords,0),torch.stack(cells,0),scales

def sample_dense_patch(volume,coord): return F.grid_sample(volume,coord,mode='bilinear',padding_mode='border',align_corners=False)
def crop_halo(x,halo):
    h=int(halo)
    return x if h<=0 else x[:,:,h:-h,h:-h,h:-h]
