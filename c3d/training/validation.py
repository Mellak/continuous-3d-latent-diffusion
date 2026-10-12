"""Fixed held-out dense patches: PSNR/SSIM describe this monitor, not full volumes."""
import random
import torch
from c3d.losses.ssim_3d import ssim3d
from c3d.visualization.ae_preview import render_region
from c3d.training.runtime import should_stop

@torch.no_grad()
def validate(model,dataset,cfg,device,run):
    v=cfg['validation'];native=tuple(cfg['experiment']['native_shape']);head=cfg['model']['head']['config'];size=tuple(min(int(p),n) for p,n in zip(v.get('patch_size',[32]*3),native))
    rng=random.Random(int(v.get('seed',91415)));sums={'l1':0.,'mse':0.,'ssim':0.};count=0;was=model.training;model.eval()
    try:
        for i in range(min(len(dataset),int(v.get('max_volumes',2)))):
            if should_stop(run):return None
            x=dataset[i]['volume'].unsqueeze(0).to(device);features,_,_=model.encode_features(x,sample_latent=False)
            for _ in range(int(v.get('patches_per_volume',4))):
                if should_stop(run):return None
                start=tuple(rng.randrange(n-p+1) for n,p in zip(native,size));end=tuple(s+p for s,p in zip(start,size))
                pred=render_region(model,features,native,start,end,int(head.get('render_chunk',65536)),int(head.get('halo',8))).float()
                target=x[(slice(None),slice(None),*(slice(s,e) for s,e in zip(start,end)))].float()
                window=min(7,min(size));window=window if window%2 else window-1
                sums['l1']+=float((pred-target).abs().mean());sums['mse']+=float((pred-target).square().mean());sums['ssim']+=float(ssim3d(pred,target,window=window));count+=1
        if not count:raise ValueError('Validation monitor contains no patches')
        import math
        result={k:value/count for k,value in sums.items()};result['psnr']=10*math.log10(4/max(result['mse'],1e-12));result['patch_count']=count
        if not all(math.isfinite(x) for x in result.values()):raise FloatingPointError('Nonfinite validation metric')
        return result
    finally:model.train(was)
