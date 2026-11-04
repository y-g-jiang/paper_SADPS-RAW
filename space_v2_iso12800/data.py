from functools import lru_cache
import json
import numpy as np
import torch
from .common import *

@lru_cache(maxsize=16)
def clean_source(path):
    p=Path(path)
    if not p.is_absolute(): p=CACHE/p
    a=np.load(p,mmap_mode='r'); return a,max(float(np.median(a)),.001)

def pack(a):
    return np.stack((a[0::2,0::2,0],a[0::2,1::2,1],a[1::2,0::2,1],a[1::2,1::2,2]))

class Scenes:
    def __init__(self,split='train',seed=7103):
        assert split in ['train','dev'],'Sealed test requires separate final-freeze entry'
        self.split=split; self.seed=seed
        self.rows=[r for r in json.loads((DATA/'MANIFEST.json').read_text())['rows'] if r['split']==split]
        self.order={}

    def auxiliary(self,rng,size,family):
        yy,xx=np.mgrid[:size,:size]/size
        if family==0: x=np.full((size,size),rng.uniform(.05,1))
        elif family==1:
            x=rng.uniform(0,.2)+rng.uniform(.2,1)*xx**rng.uniform(.5,2)+rng.uniform(.1,.8)*yy**rng.uniform(.5,2)
        elif family==2:
            angle=rng.uniform(0,2*np.pi); coordinate=xx*np.cos(angle)+yy*np.sin(angle)
            location=rng.uniform(coordinate.min(),coordinate.max()); band=rng.uniform(.005,.08)
            mask=coordinate>location if rng.random()<.5 else abs(coordinate-location)<band
            x=np.where(mask,rng.uniform(.4,1),rng.uniform(0,.2))
        elif family in [3,4]:


            width=size//8; small=rng.uniform(.02,1,(16,16))
            indices=np.minimum(np.arange(width)*16//width,15); motif=small[indices[:,None],indices[None,:]]
            x=np.full((size,size),rng.uniform(.02,.15)); center=(size-width)//2
            x[center:center+width,center:center+width]=motif
            for top,left in [(size//16,size//16),(size//16,13*size//16),(13*size//16,size//16),(13*size//16,13*size//16)]:
                shuffled=rng.permutation(motif.ravel()).reshape(motif.shape)
                x[top:top+width,left:left+width]=motif if family==3 else shuffled
        else:
            x=np.full((size,size),rng.uniform(.2,.5))
            for _ in range(8):
                angle=rng.uniform(0,2*np.pi); frequency=np.exp(rng.uniform(np.log(5),np.log(200)))
                x+=rng.uniform(.01,.08)*np.sin(frequency*(xx*np.cos(angle)+yy*np.sin(angle))+rng.uniform(0,6.28))
            x=np.maximum(x,0)
        return np.repeat(x[...,None],3,axis=2).astype(np.float32)

    def sample(self,ordinal,context=128,force_natural=False,family=None):
        offset=0 if self.split=='train' else 100_000_000
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,offset,ordinal]))
        aux=family is not None or (self.split=='train' and not force_natural and ordinal%10==9)

        full_size=max(256,context); raw_size=2*full_size
        if aux:
            fam=int(family if family is not None else (ordinal//10)%6)
            a=self.auxiliary(rng,raw_size,fam); median=max(float(np.median(a)),.001)
            scene=f'S{fam}_{offset}_{ordinal}'; domain=f'S{fam}'; valid=np.ones((raw_size,raw_size),np.float32)
        else:
            position=ordinal if force_natural or self.split!='train' else ordinal-ordinal//10
            epoch,index=divmod(position,len(self.rows))
            if self.split=='train':
                if epoch not in self.order: self.order={epoch:np.random.default_rng(self.seed+epoch+900000).permutation(len(self.rows))}
                index=int(self.order[epoch][index])
            row=self.rows[index]; base,median=clean_source(row['cache_path'])
            h,w=base.shape[:2]

            cy=int(rng.integers(min(64,h//2),max(min(64,h//2)+1,h-min(64,h//2))))//2*2
            cx=int(rng.integers(min(64,w//2),max(min(64,w//2)+1,w-min(64,w//2))))//2*2
            top=cy-raw_size//2; left=cx-raw_size//2
            y0=max(0,top); y1=min(h,top+raw_size); x0=max(0,left); x1=min(w,left+raw_size)
            a=np.zeros((raw_size,raw_size,3),np.float32); valid=np.zeros((raw_size,raw_size),np.float32)
            a[y0-top:y1-top,x0-left:x1-left]=base[y0:y1,x0:x1]
            valid[y0-top:y1-top,x0-left:x1-left]=1
            scene=row['scene_group']; domain='natural'

        rotation=int(rng.integers(4)); a=np.rot90(a,rotation); valid=np.rot90(valid,rotation)
        if rng.random()<.5: a=a[:,::-1]; valid=valid[:,::-1]
        level=float(np.exp(rng.uniform(np.log(.4),np.log(64))))
        color=np.exp(rng.uniform(-.3,.3,3)).astype(np.float32)
        signal=np.asarray(a*(level/median)*color,np.float32)
        if rng.random()<.3: signal+=rng.uniform(.1,2,3).astype(np.float32)
        black=bool(rng.random()<.05)
        if black: signal[:]=0
        packed=pack(signal); vm=pack(np.repeat(valid[...,None],3,2)).min(0,keepdims=True)
        b=rng.uniform(-.25,.25,(4,1,1)).astype(np.float32)

        z=G*rng.poisson(packed/G)+rng.normal(0,RN,packed.shape)+b
        adc=np.floor(BLACK+z+.5)
        assert np.all(adc>=0), 'ADC lower floor reached:512 headroom contract violated'
        v=(np.minimum(adc,16383)-BLACK).astype(np.int16); y=np.maximum(v,0).astype(np.float32)

        y*=vm
        off=(full_size-context)//2; sl=(slice(None),slice(off,off+context),slice(off,off+context))
        return dict(y=torch.from_numpy(y[sl].copy()),s=torch.from_numpy(packed[sl].copy()),
            v=torch.from_numpy(v[sl].copy()),z_gt=torch.from_numpy(z[sl].copy()),lambda_gt=torch.from_numpy((packed[sl]/G).copy()),candidate_zero_mask=torch.from_numpy(((y[sl]==0)&(vm[sl]>0)).copy()),actual_negative_mask=torch.from_numpy((v[sl]<0).copy()),adc_raw_gt=torch.from_numpy((v[sl].astype(np.int32)+BLACK).copy()),b=torch.from_numpy(b.copy()),valid=torch.from_numpy(vm[sl].copy()),
            scene=scene,domain=domain,ordinal=ordinal,level=level,black=black)

def core_mask(case,core=64):
    y=case['y']; h,w=y.shape[-2:]; mask=torch.zeros_like(y,dtype=torch.bool)
    a=(h-core)//2; b=(w-core)//2
    mask[:,a:a+core,b:b+core]=True
    return mask&(y==0)&(case['valid']>0)

def to_device(case,device='cuda'):
    return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in case.items()}
