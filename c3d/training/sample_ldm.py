#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np,torch,yaml
from c3d.models.ldm.model import build_ldm
from c3d.models.ldm.ema import EMA
from c3d.diffusion.samplers.heun import sample as jit_sample
from c3d.diffusion.ddpm import make_schedule,sample as ddpm_sample
from c3d.models.ae.autoencoder_3d import build_autoencoder,load_checkpoint_state,remap_legacy_state
from c3d.data.latent_dataset import denormalize
from c3d.training.runtime import autocast
from c3d.visualization.volume_slices import save_generated_three_planes

def load_yaml(p):return yaml.safe_load(os.path.expandvars(Path(p).read_text()))
def load_ae(manifest,device):
    from c3d.data.identity import verify_ae_manifest
    from c3d.models.ae.autoencoder_3d import load_complete_model
    cfg=verify_ae_manifest(manifest)
    return load_complete_model(cfg,manifest['ae_checkpoint'],device),cfg

def sample(checkpoint,out_dir,num=1,seed=91415,steps=None,method=None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');c=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=c['config'];stats=c['latent_stats'];manifest=c['latent_manifest'];obj=cfg['objective'];tr=cfg['training']
    from c3d.data.identity import digest
    if c.get('latent_identity')!=digest({'manifest':manifest,'stats':stats}):raise ValueError('LDM latent/stats identity mismatch')
    mc=dict(cfg['model']);shape=tuple(int(v) for v in stats['latent_shape']);mc['in_channels']=shape[0];mc['out_channels']=shape[0]
    if c.get('schema',2)<3:mc.setdefault('decoder_skip','concat')
    model=build_ldm(mc,obj,stats).to(device).eval();model.load_state_dict(c['model'],strict=True);ema=EMA(model,float(tr.get('ema_decay',.9999)));ema.load_state_dict(c['ema'])
    del c  # Optimizer/resume tensors are unnecessary for generation.
    ae,ae_cfg=load_ae(manifest,device);native=tuple(int(v) for v in manifest['native_shape']);hc=ae_cfg['model']['head']['config'];out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    precision=tr.get('precision','bf16');amp=device.type=='cuda';base_seed=int(seed)
    if obj['name'].startswith('ddpm'):
        if steps is not None or method is not None:raise ValueError('DDPM sampling uses the configured discrete schedule; --steps/--method are JiT-only')
        steps=int(obj.get('num_steps',1000));method='ddpm'
    else:
        steps=int(steps or tr.get('sample_steps',50));method=method or tr.get('sampling_method','heun')
    with ema.applied(model),torch.no_grad():
        for i in range(int(num)):
            this_seed=base_seed+i
            with autocast(device,amp,precision):
                if obj['name']=='jit_x0v':zn=jit_sample(model,(1,*shape),obj,steps,method,this_seed,device)
                else:
                    sch=make_schedule(int(obj.get('num_steps',1000)),float(obj.get('beta_start',1e-4)),float(obj.get('beta_end',2e-2)),obj.get('schedule','linear'))
                    zn=ddpm_sample(model,(1,*shape),sch,'x0' if obj['name']=='ddpm_x0' else 'eps',this_seed,device)
                zr=denormalize(zn.float(),stats);vol=ae.decode_latent(zr,native,chunk=int(hc.get('render_chunk',300000)),tile_size=tuple(hc.get('render_tile',[32,64,64])),halo=int(hc.get('halo',8)))
            np.save(out/f'sample_{i:04d}_latent_normalized.npy',zn[0].float().cpu().numpy());np.save(out/f'sample_{i:04d}_latent_raw.npy',zr[0].float().cpu().numpy());np.save(out/f'sample_{i:04d}_volume.npy',vol[0,0].float().cpu().numpy())
            save_generated_three_planes(vol,out/f'sample_{i:04d}.png',f'{obj["name"]} seed={this_seed}')
            del zn,zr,vol
            if device.type=='cuda':torch.cuda.empty_cache()
    (out/'generation.json').write_text(json.dumps({'checkpoint':str(Path(checkpoint).resolve()),'objective':obj['name'],'num':int(num),'seed':base_seed,'steps':steps,'method':method,'ae_checkpoint':manifest['ae_checkpoint']},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--num',type=int,default=1);p.add_argument('--seed',type=int,default=91415);p.add_argument('--steps',type=int);p.add_argument('--method',choices=['heun','euler']);a=p.parse_args();sample(a.checkpoint,a.out_dir,a.num,a.seed,a.steps,a.method)
if __name__=='__main__':main()
