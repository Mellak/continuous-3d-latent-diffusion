from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

def _norm(x):return ((x.detach().float().cpu().clamp(-1,1)+1)*.5).numpy()
def save_three_plane_comparison(target,recon,path,title=''):
    """3 rows (axial/coronal/sagittal), columns target/recon/error."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t=target[0,0].detach().float().cpu();r=recon[0,0].detach().float().cpu()
    d,h,w=t.shape;planes=[(t[d//2],r[d//2]),(t[:,h//2,:],r[:,h//2,:]),(t[:,:,w//2],r[:,:,w//2])]
    fig,ax=plt.subplots(3,3,figsize=(10,10));names=['axial','coronal','sagittal']
    for i,(a,b) in enumerate(planes):
        aa=((a.clamp(-1,1)+1)*.5).numpy();bb=((b.clamp(-1,1)+1)*.5).numpy();ee=np.abs(aa-bb)
        for j,img in enumerate((aa,bb,ee)):ax[i,j].imshow(img,cmap='gray',vmin=0,vmax=1);ax[i,j].axis('off')
        ax[i,0].set_title(f'{names[i]} target');ax[i,1].set_title('recon');ax[i,2].set_title('abs error')
    if title:fig.suptitle(title)
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
def save_generated_three_planes(volume,path,title=''):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);x=volume[0,0].detach().float().cpu();d,h,w=x.shape
    imgs=[x[d//2],x[:,h//2,:],x[:,:,w//2]];fig,ax=plt.subplots(1,3,figsize=(12,4))
    for a,img,n in zip(ax,imgs,['axial','coronal','sagittal']):a.imshow(((img.clamp(-1,1)+1)*.5).numpy(),cmap='gray',vmin=0,vmax=1);a.axis('off');a.set_title(n)
    if title:fig.suptitle(title)
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
