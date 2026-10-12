from __future__ import annotations
from pathlib import Path
import shutil,torch

def atomic_torch_save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);tmp.replace(path)
def save_ae(run_dir,payload,is_best=False,epoch=None,keep_every=50):
    # epoch/keep_every remain accepted for callers from older versions.
    # AE runs retain only last.pt and best.pt; never grow periodic archives.
    run=Path(run_dir);ck=run/'checkpoints';ck.mkdir(parents=True,exist_ok=True)
    pending=ck/'last.pt.tmp';torch.save(payload,pending)
    if is_best:
        temporary=ck/'best.pt.tmp';shutil.copyfile(pending,temporary);temporary.replace(ck/'best.pt')
    pending.replace(ck/'last.pt')
def save_ldm(run_dir,payload,is_best=False,step=None,keep_every=0):
    # Legacy arguments are accepted, but numbered archives are never created.
    # Publish best first: if interrupted between renames, last remains older
    # and will replay validation instead of losing the selected best state.
    run=Path(run_dir);ck=run/'checkpoints';ck.mkdir(parents=True,exist_ok=True)
    pending=ck/'last.pt.tmp'
    with pending.open('wb') as handle:
        torch.save(payload,handle);handle.flush()
        import os
        os.fsync(handle.fileno())
    if is_best:
        temporary=ck/'best.pt.tmp';shutil.copyfile(pending,temporary);temporary.replace(ck/'best.pt')
    pending.replace(ck/'last.pt')
def find_last(run_dir):
    p=Path(run_dir)/'checkpoints'/'last.pt';return p if p.exists() else None
