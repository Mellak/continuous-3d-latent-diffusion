"""Render only the three displayed slices; bounded CLIF tiles include their halo."""
import torch
from c3d.models.heads.liif_3d import cell_3d

def render_region(model,features,spatial,start,end,chunk=65536,halo=8):
    shape=tuple(end[i]-start[i] for i in range(3));device=features.device
    if model.head_name=='clif':
        hs=tuple(max(0,start[i]-halo) for i in range(3));he=tuple(min(spatial[i],end[i]+halo) for i in range(3))
        required=(2+2*model.head.cfg.num_res_blocks)*(model.head.cfg.kernel_size//2)
        if halo<required or model.head.cfg.norm not in ('channel','none'):raise ValueError('Slice rendering requires tile-local CLIF normalization and sufficient halo')
        co,ce=model.head.make_region_coord_cell(spatial,hs,he,device=device,dtype=torch.float32,batch=features.shape[0])
        pred=model.query(features,co,ce)
        return pred[(slice(None),slice(None),*(slice(start[i]-hs[i],end[i]-hs[i]) for i in range(3)))]
    axes=[(torch.arange(start[i],end[i],device=device,dtype=torch.float32)+.5)*2/spatial[i]-1 for i in range(3)]
    zz,yy,xx=torch.meshgrid(*axes,indexing='ij');co=torch.stack((xx,yy,zz),-1).reshape(1,-1,3).expand(features.shape[0],-1,-1)
    ce=cell_3d(spatial,device,torch.float32).expand(features.shape[0],co.shape[1],3)
    parts=[model.query(features,co[:,s:s+chunk],ce[:,s:s+chunk]) for s in range(0,co.shape[1],chunk)]
    return torch.cat(parts,1).transpose(1,2).reshape(features.shape[0],1,*shape)

@torch.no_grad()
def render_planes(model,features,spatial,chunk=65536,clif_tile=(32,64,64),halo=8,stop=None):
    planes=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis];out=torch.empty((spatial[other[0]],spatial[other[1]]),dtype=torch.float32)
        for u in range(0,spatial[other[0]],clif_tile[other[0]]):
            for v in range(0,spatial[other[1]],clif_tile[other[1]]):
                if stop and stop():return None
                start=[0,0,0];end=list(spatial);start[axis]=spatial[axis]//2;end[axis]=start[axis]+1
                start[other[0]]=u;end[other[0]]=min(u+clif_tile[other[0]],spatial[other[0]])
                start[other[1]]=v;end[other[1]]=min(v+clif_tile[other[1]],spatial[other[1]])
                p=render_region(model,features,spatial,start,end,chunk,halo)[0,0].squeeze(axis)
                out[u:end[other[0]],v:end[other[1]]]=p.float().cpu()
        planes.append(out)
    return planes

def save_planes(planes,path,targets=None,title=''):
    from pathlib import Path
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    cols=3 if targets is not None else 1;fig,axes=plt.subplots(3,cols,figsize=(4*cols,10),squeeze=False)
    for i,p in enumerate(planes):
        pred=(p.float().cpu().numpy()+1)/2
        images=[pred] if targets is None else [(targets[i].float().cpu().numpy()+1)/2,pred,np.abs(targets[i].float().cpu().numpy()-p.float().cpu().numpy())/2]
        for j,img in enumerate(images):axes[i,j].imshow(img,cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off')
        axes[i,0].set_title(('axial','coronal','sagittal')[i])
    if targets is not None:
        for i,plane in enumerate(('Axial','Coronal','Sagittal')):
            for j,label in enumerate(('Target','Reconstruction','Absolute error')):axes[i,j].set_title(f'{plane} | {label}')
    temporary=path.with_name(path.stem+'.tmp'+path.suffix)
    try:
        fig.suptitle(title);fig.tight_layout();fig.savefig(temporary,dpi=120);temporary.replace(path)
    finally:plt.close(fig)

@torch.no_grad()
def save_ae_preview(model,volume,spatial,path,chunk=65536,clif_tile=(32,64,64),halo=8,stop=None,title=None):
    was=model.training;model.eval()
    try:
        features,_,_=model.encode_features(volume,sample_latent=False)
        planes=render_planes(model,features,spatial,chunk,clif_tile,halo,stop)
        if planes is None:return False
        x=volume[0,0].float().cpu();d,h,w=spatial
        save_planes(planes,path,[x[d//2],x[:,h//2,:],x[:,:,w//2]],title or f'{model.head_name}: held-out reconstruction')
        return True
    finally:model.train(was)
