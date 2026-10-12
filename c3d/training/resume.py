"""Epoch permutation/cursor and per-rank RNG state for restartable training."""
import random, math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

class EpochBatches(Sampler):
    def __init__(self,size,batch,rank,world,seed,epoch,cursor=0):
        self.size=size;self.batch=batch;self.rank=rank;self.world=world;self.cursor=cursor
        if not 0<=cursor<=size:raise ValueError('Invalid sample cursor')
        self.order=torch.randperm(size,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    def __iter__(self):
        stride=self.batch*self.world
        for start in range(self.cursor,self.size,stride):
            ids=self.order[start:start+stride]
            ids += [self.order[i%self.size] for i in range(stride-len(ids))]
            yield ids[self.rank*self.batch:(self.rank+1)*self.batch]
    def __len__(self):return math.ceil((self.size-self.cursor)/(self.batch*self.world))

def loader_for_epoch(ds,tr,rank,world,epoch,cursor=0):
    seed=int(tr.get('seed',0));ds.epoch=epoch;ds.augmentation_seed=seed
    batch=int(tr.get('batch_size',tr.get('batch_per_gpu',1)))
    # A single Slurm task supplies the CPU budget for all ranks.
    import os
    cpus=int(os.getenv('SLURM_CPUS_PER_TASK',str(max(1,world))))
    workers=min(int(tr.get('num_workers',tr.get('workers_per_rank',0))),max(0,cpus//world-1))
    sampler=EpochBatches(len(ds),batch,rank,world,seed,epoch,cursor)
    return DataLoader(ds,batch_sampler=sampler,num_workers=workers,persistent_workers=False,pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(seed+epoch)),batch

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state(s['cuda'].cpu())

def gather_rng():
    local=capture_rng()
    if not dist.is_initialized():return [local]
    states=[None]*dist.get_world_size();dist.all_gather_object(states,local);return states

def restore_training_rng(checkpoint,rank,world,seed):
    states=checkpoint.get('rng_states')
    if not states or 'cursor' not in checkpoint:raise ValueError('Checkpoint lacks resumable RNG/cursor state; use it as initialization in a NEW run')
    if int(checkpoint.get('world_size',len(states)))==world:restore_rng(states[rank])
    else:
        from c3d.training.runtime import seed_everything
        seed_everything(int(seed)+int(checkpoint.get('global_step',checkpoint.get('step',0)))*100003+rank)
        if rank==0:print('[RESUME] GPU count changed: remaining sample order retained, RNG reseeded; global batch changes with world size.',flush=True)
