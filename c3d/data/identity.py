"""Content identities and atomic files shared by caches and checkpoints."""
import hashlib, json, os
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic_json(path,obj):
    if Path(path).name=="status.json" and isinstance(obj,dict):obj={**obj,"job_id":obj.get("job_id",os.getenv("SLURM_JOB_ID","local"))}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def source_records(files):
    return [{'path':str(Path(p).resolve()),'sha256':sha256(p),'bytes':Path(p).stat().st_size} for p in files]

def verify_ae_manifest(manifest):
    p=Path(manifest['ae_checkpoint'])
    if sha256(p)!=manifest['ae_checkpoint_sha256']:
        raise ValueError('AE checkpoint content changed: decoder identity mismatch')
    cfg=manifest.get('ae_config_resolved')
    if cfg is None:raise ValueError('Cache lacks immutable resolved AE config; rebuild the cache')
    if digest(cfg)!=manifest['ae_config_sha256']:raise ValueError('AE config identity mismatch')
    return cfg
