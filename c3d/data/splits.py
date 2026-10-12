"""Reproducible disjoint train/validation manifests, shared with latent extraction."""
import copy, random
from pathlib import Path
from c3d.data.identity import digest, atomic_json

def split_files(files, config):
    files=sorted(str(Path(p).resolve()) for p in files)
    v=config.get('validation',{}); explicit=v.get('file_list')
    if explicit:
        val=[str(Path(x.strip()).expanduser().resolve()) for x in Path(explicit).read_text().splitlines() if x.strip() and not x.startswith('#')]
        if not val or any(not Path(x).is_file() for x in val):raise ValueError('Invalid validation file list')
        train=[x for x in files if x not in set(val)]
    else:
        fraction=float(v.get('fraction',0.1))
        if not 0<fraction<1 or len(files)<2:raise ValueError('Held-out validation needs at least two volumes and 0 < fraction < 1, or an explicit validation file_list')
        shuffled=files.copy();random.Random(int(v.get('seed',91415))).shuffle(shuffled)
        n=max(1,min(len(files)-1,round(len(files)*fraction)));val=sorted(shuffled[:n]);train=sorted(shuffled[n:])
    if not train or not val or set(train)&set(val):raise ValueError('Train/validation split must be nonempty and disjoint')
    identity={'train':train,'validation':val,'validation_config':v,'source_metadata':[{'path':x,'bytes':Path(x).stat().st_size,'mtime_ns':Path(x).stat().st_mtime_ns} for x in sorted(set(train+val))]}
    return train,val,digest(identity)

def split_datasets(dataset,config,run=None):
    train,val,identity=split_files(dataset.files,config)
    if run is not None:
        import json
        p=Path(run)/'split.json';obj={'train':train,'validation':val,'identity':identity}
        if p.exists() and json.loads(p.read_text())!=obj:raise ValueError('Dataset split changed for this run')
        atomic_json(p,obj)
    a=copy.copy(dataset);b=copy.copy(dataset);a.files=list(map(Path,train));b.files=list(map(Path,val));b.augment_flip=False
    return a,b,identity
