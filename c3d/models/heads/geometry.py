"""FP32 geometry with dtype-preserving nearest-neighbor feature gathers."""
import torch
import torch.nn.functional as F

def nearest_sample_3d(field,grid,padding_mode='border',align_corners=False):
    """grid_sample(nearest) semantics without a full field.float() allocation.

    Input [B,C,D,H,W], grid [B,Do,Ho,Wo,3]. XYZ indices are calculated
    in FP32, using round-to-even like grid_sample. Gather output and its
    feature gradient retain the field dtype (including BF16).
    """
    if field.ndim!=5 or grid.ndim!=5 or grid.shape[-1]!=3:
        raise ValueError('Nearest 3D sampling expects a 5D field and XYZ grid')
    if field.shape[0]!=grid.shape[0]:raise ValueError('Sampling batch mismatch')
    if padding_mode not in ('border','zeros','reflection'):raise ValueError(padding_mode)
    shape=grid.shape[1:4];b,c,d,h,w=field.shape
    xyz=grid.float().reshape(b,-1,3)
    indices=[];valid=torch.ones(xyz.shape[:2],device=field.device,dtype=torch.bool)
    for axis,size in enumerate((w,h,d)):
        coord=xyz[...,axis]
        coord=torch.where(torch.isnan(coord),coord.new_tensor(-1.),coord)
        pos=((coord+1)*(size-1)/2 if align_corners else ((coord+1)*size-1)/2)
        if padding_mode=='reflection':
            low=0. if align_corners else -.5
            span=float(size-1 if align_corners else size)
            if span==0:pos=torch.zeros_like(pos)
            else:
                distance=(pos-low).abs();flips=torch.floor(distance/span)
                extra=torch.remainder(distance,span)
                pos=torch.where(torch.remainder(flips,2)==0,extra,span-extra)+low
        if padding_mode!='zeros':pos=pos.clamp(0,size-1)
        rounded=torch.round(pos)
        valid &= torch.isfinite(rounded)&(rounded>=0)&(rounded<size)
        indices.append(torch.nan_to_num(rounded,nan=0.,posinf=size-1,neginf=0.).clamp(0,size-1).long())
    ix,iy,iz=indices;index=iz*(h*w)+iy*w+ix
    values=torch.gather(field.flatten(2),2,index.unsqueeze(1).expand(-1,c,-1))
    if padding_mode=='zeros':values=values*valid.unsqueeze(1).to(values.dtype)
    # Nearest interpolation has zero coordinate derivatives, not a missing
    # gradient; keep this behavior for callers with trainable finite grids.
    if grid.requires_grad:
        zero=torch.nan_to_num(xyz).sum(-1).mul(0).to(values.dtype)
        values=values+zero.unsqueeze(1)
    return values.reshape(b,c,*shape)

def sample_fp32(field, grid, **kwargs):
    # Historical public name retained for imports. Only small interpolation
    # paths (ground-truth sampling) still require FP32 fields.
    if kwargs.get('mode','bilinear')=='nearest' and field.ndim==5:
        return nearest_sample_3d(field,grid,kwargs.get('padding_mode','zeros'),kwargs.get('align_corners',False))
    with torch.autocast(field.device.type, enabled=False):
        return F.grid_sample(field.float(), grid.float(), **kwargs)
