from __future__ import annotations
import csv,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def seed_everything(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def setup_distributed(seed=0):
    world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'));distributed=world>1
    if torch.cuda.is_available():
        device=torch.device('cuda',local);torch.cuda.set_device(device)
    else:device=torch.device('cpu')
    if distributed and not dist.is_initialized():
        from datetime import timedelta
        kwargs={'device_id':device} if device.type=='cuda' else {}
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=int(os.getenv('DDP_TIMEOUT_SECONDS','7200'))),**kwargs)
    torch.set_num_threads(1)
    seed_everything(seed+rank);return distributed,rank,local,world,device
def wrap_ddp(m,device,distributed):return DDP(m,device_ids=[device.index],output_device=device.index,find_unused_parameters=False) if distributed and device.type=='cuda' else (DDP(m) if distributed else m)
def unwrap(m):return m.module if hasattr(m,'module') else m
def is_main(rank):return rank==0
def barrier():
    if dist.is_initialized():
        if dist.get_backend()=='nccl':dist.barrier(device_ids=[torch.cuda.current_device()])
        else:dist.barrier()
def cleanup():
    if dist.is_initialized():dist.destroy_process_group()
def autocast(device,enabled,dtype='fp16'):
    dt=torch.float16 if dtype=='fp16' else torch.bfloat16
    return torch.autocast(device_type=device.type,dtype=dt,enabled=bool(enabled and device.type=='cuda' and dtype in ('fp16','bf16')))
def scaler(device,enabled,dtype):return torch.amp.GradScaler('cuda',enabled=bool(enabled and dtype=='fp16' and device.type=='cuda'))
def set_requires_grad(m,value):
    if m is None:return
    for p in m.parameters():p.requires_grad_(value)
def parameter_count(m,trainable_only=False):return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))
def append_csv(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row.keys()));
        if not exists:w.writeheader()
        w.writerow(row)
def atomic_json(path,obj):
    if Path(path).name=="status.json" and isinstance(obj,dict):obj={**obj,"job_id":obj.get("job_id",os.getenv("SLURM_JOB_ID","local"))}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
def seconds_left():
    deadline=float(os.getenv('TRAIN_DEADLINE_UNIX','inf'))
    return deadline-time.time()

def should_stop(run_dir):
    job=os.getenv('SLURM_JOB_ID','local')
    root=Path(os.getenv('C3D_RUN_CONTROL_DIR',str(Path(run_dir)/'control')))
    return ((root/'STOP').exists() or (root/f'stop_{job}').exists() or (Path(run_dir)/'control'/'STOP').exists()
            or (Path(run_dir)/'control'/f'stop_{job}').exists()
            or seconds_left() <= float(os.getenv('SAVE_MARGIN_SECONDS','300')))

def coordinated_stop(run_dir,device):
    flag=torch.tensor(int(should_stop(run_dir)),device=device)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag.item())
def reduce_metrics(sums,count,device):
    names=list(sums);v=torch.tensor([float(sums[k]) for k in names]+[float(count)],device=device,dtype=torch.float64)
    if dist.is_initialized():dist.all_reduce(v)
    n=max(1.0,float(v[-1]));return {k:float(v[i]/n) for i,k in enumerate(names)}
def system_info():
    info={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'hostname':os.uname().nodename,'time':time.time()}
    if torch.cuda.is_available(): info['gpus']=[{'name':torch.cuda.get_device_name(i),'total_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]
    return info
