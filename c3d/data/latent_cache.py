#!/usr/bin/env python3
"""Restartable raw latent cache with CPU statistics and online normalization."""
from __future__ import annotations
import argparse,json,os,traceback
from pathlib import Path
import numpy as np
import torch,yaml
from c3d.models.ae.autoencoder_3d import load_complete_model
from c3d.data.medical_volume import MedicalVolumeDataset3D,parse_float3
from c3d.data.splits import split_datasets
from c3d.data.identity import sha256,digest,atomic_json,source_records
from c3d.training.runtime import setup_distributed,is_main,barrier,cleanup,coordinated_stop

def load_cfg(path):return yaml.safe_load(os.path.expandvars(Path(path).read_text()))
def dataset_from_cfg(cfg):
    d=cfg['data'];e=cfg['experiment']
    return MedicalVolumeDataset3D(root=d['root'],modality=e['modality'],target_shape=tuple(e['native_shape']),target_spacing_xyz=parse_float3(d.get('target_spacing')),patterns=d.get('patterns'),file_list=d.get('file_list'),augment_flip=False,
        ct_min_hu=d.get('ct_min_hu',-1000),ct_max_hu=d.get('ct_max_hu',1500),ct_pad_hu=d.get('ct_pad_hu',-1000),mri_percentile_low=d.get('mri_percentile_low',1),mri_percentile_high=d.get('mri_percentile_high',99),mri_foreground_threshold=d.get('mri_foreground_threshold',0))
def load_model(cfg,checkpoint,device):return load_complete_model(cfg,checkpoint,device)

from c3d.data.latent_statistics import save_array, valid_array, streaming_stats, finalize

def prepare(ae_config,checkpoint,out_root,normalization='global_sigma',keep_raw=True):
    if not keep_raw:raise ValueError('--drop-raw is incompatible with on-the-fly normalization; raw latents are the training input')
    cfg=load_cfg(ae_config);_,rank,_,world,device=setup_distributed(0);out=Path(out_root);raw=out/'raw';norm=out/'normalized'
    try:
        if is_main(rank):
            raw.mkdir(parents=True,exist_ok=True);(out/'control').mkdir(exist_ok=True)
        barrier();all_ds=dataset_from_cfg(cfg);ds,_,split_id=split_datasets(all_ds,cfg)
        if is_main(rank):
            import shutil
            checkpoint_hash=sha256(checkpoint);snapshot=out/'ae_checkpoint.pt'
            if not snapshot.exists():
                temporary=snapshot.with_suffix('.tmp');shutil.copyfile(checkpoint,temporary)
                if sha256(temporary)!=checkpoint_hash:raise ValueError('AE checkpoint changed while copying; select a stable checkpoint')
                temporary.replace(snapshot)
            if sha256(snapshot)!=checkpoint_hash:raise ValueError('Cache AE checkpoint identity changed; use a new output directory')
            manifest={'schema':2,'ae_config':str(Path(ae_config).resolve()),'ae_config_resolved':cfg,'ae_config_sha256':digest(cfg),'ae_checkpoint':str(snapshot.resolve()),'ae_checkpoint_source':str(Path(checkpoint).resolve()),'ae_checkpoint_sha256':checkpoint_hash,
                'head':cfg['model']['head']['name'],'modality':cfg['experiment']['modality'],'native_shape':cfg['experiment']['native_shape'],'number_of_latents':len(ds),'normalization':normalization,'split_identity':split_id,'sources':source_records(ds.files)}
            mp=out/'manifest.json'
            if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Cache AE/config/source/split/normalization identity changed; use a new cache directory')
            if not mp.exists() and (list(raw.glob('*.npy')) or list(norm.glob('*.npy'))):raise ValueError('Unidentified old cache; use a new cache directory')
            atomic_json(mp,manifest);(out/'READY.json').unlink(missing_ok=True);atomic_json(out/'control/status.json',{'status':'running'})
        barrier();manifest=json.loads((out/'manifest.json').read_text());identity=digest(manifest);m=load_model(cfg,manifest["ae_checkpoint"],device)
        with torch.no_grad():
            for start in range(0,len(ds),world):
                if coordinated_stop(out,device):
                    if is_main(rank):atomic_json(out/'control/status.json',{'status':'paused'})
                    return
                i=start+rank
                if i>=len(ds):continue
                path=raw/f'latent_{i:08d}.npy'
                if valid_array(path,identity):continue
                x=ds[i]['volume'].unsqueeze(0).to(device);z,_=m.encode(x,sample=False)
                save_array(path,z[0].float().cpu().numpy(),identity);del x,z
        del m;barrier()
        if is_main(rank):finalize(out)
        barrier()
    except Exception:
        if is_main(rank):atomic_json(out/'control/status.json',{'status':'failed','error':traceback.format_exc()})
        raise
    finally:cleanup()

def main():
    p=argparse.ArgumentParser();p.add_argument('--ae-config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out-root',required=True);p.add_argument('--normalization',choices=['global_sigma','zscore'],default='global_sigma');p.add_argument('--drop-raw',action='store_true');a=p.parse_args();prepare(a.ae_config,a.checkpoint,a.out_root,a.normalization,not a.drop_raw)
if __name__=='__main__':main()
