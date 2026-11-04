import math
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from scipy.special import logsumexp
from .common import G,RN,K,UPPER
from . import baselines as legacy
from space_v2.context_baselines import histogram_at,refined_mle
from baselines.local_pg import prior_weights
from space_v2.fast_pg import adaptive

def install_exact_gpu():

    from . import pg
    original=pg.exact
    @torch.no_grad()
    def exact_gpu(s,b):
        ss,bb=torch.broadcast_tensors(s.double(),b.double());shape=ss.shape


        if s.requires_grad or b.requires_grad or not bool((bb==bb.reshape(-1)[0]).all()):return original(s,b)
        target=s.device;signals=ss.reshape(-1);bias=float(bb.reshape(-1)[0]);parts=[]
        for start in range(0,len(signals),512):
            lam=signals[start:start+512].to('cuda')/G
            den,_=adaptive(lam,.5,bias,G,RN,'cdf',device='cuda',radius=16)
            q=torch.empty((len(lam),K+2),dtype=torch.float64,device='cuda')
            for j in range(0,K+1,64):
                codes=-torch.arange(j,min(j+64,K+1),dtype=torch.float64,device='cuda')[None]
                values,_=adaptive(lam[:,None],codes,bias,G,RN,'bin',device='cuda',radius=16)
                q[:,j:j+codes.shape[1]]=values-den[:,None]
            tail,_=adaptive(lam,-K-.5,bias,G,RN,'cdf',device='cuda',radius=16)
            q[:,-1]=tail-den;parts.append(q.to(target))
        return torch.cat(parts).reshape(*shape,K+2)
    pg.exact=exact_gpu

@torch.no_grad()
def local_laws(y,valid,mask,b,prior,decoder):
    yy=np.asarray(y,np.int64);valid=np.asarray(valid,bool);centers=np.argwhere(mask)
    means=uniform_filter(yy.astype(float)*valid,3,mode='constant')/np.maximum(uniform_filter(valid.astype(float),3,mode='constant'),1e-12)
    end=max(128.,1.8*float(means[mask].max())+6*RN)
    grid=np.unique(np.r_[np.arange(0,32,.1),np.arange(32,end,.5),end])/G
    codes=np.unique(yy[valid]);hist=histogram_at(yy,valid,centers,3,codes)
    hg=torch.as_tensor(hist,dtype=torch.float64,device='cuda')
    likelihood=np.empty((len(centers),len(grid)),np.float64);maximum=-math.inf



    for start in range(0,len(grid),512):
        gg=torch.as_tensor(grid[start:start+512],dtype=torch.float64,device='cuda')[:,None]
        ll=torch.empty((len(gg),len(codes)),dtype=torch.float64,device='cuda')
        bins=np.flatnonzero((codes>0)&(codes!=UPPER))
        for j in range(0,len(bins),64):
            cols=bins[j:j+64];ks=torch.as_tensor(codes[cols],dtype=torch.float64,device='cuda')[None]
            values,bound=adaptive(gg,ks,b,G,RN,'bin',device='cuda',radius=16)
            ll[:,torch.as_tensor(cols,device='cuda')]=values;maximum=max(maximum,bound)
        for code,threshold,event in [(0,.5,'cdf'),(UPPER,UPPER-.5,'upper')]:
            cols=np.flatnonzero(codes==code)
            if len(cols):
                values,bound=adaptive(gg[:,0],threshold,b,G,RN,event,device='cuda',radius=16)
                ll[:,int(cols[0])]=values;maximum=max(maximum,bound)
        likelihood[:,start:start+len(gg)]=(hg@ll.T).cpu().numpy()
    del hg,ll
    lw=likelihood+np.log(prior_weights(grid,prior));lw-=logsumexp(lw,axis=1)[:,None]
    weights=np.exp(lw);boundary=float(weights[:,-2:].sum(1).max());assert boundary<1e-5
    kernels=legacy.decoder_numpy(decoder,grid*G,b)
    kg=torch.as_tensor(np.exp(kernels.astype(np.float64)),device='cuda')
    mixture=np.empty((len(centers),K+2),np.float64)
    for start in range(0,len(weights),512):
        mixture[start:start+512]=(torch.as_tensor(weights[start:start+512],device='cuda')@kg).cpu().numpy()
    assert (mixture>0).all();lb=np.log(mixture);lb-=logsumexp(lb,axis=1)[:,None]
    lam=refined_mle(grid,likelihood);local=legacy.decoder_numpy(decoder,lam*G,b)
    return {'LB3':lb,'L3':local},dict(grid_points=len(grid),max_signal=float(grid[-1]*G),boundary=boundary,
        backend='FP64 adaptive GPU likelihood and matrix arithmetic, unchanged grid/estimator',max_log_relative_omitted_mass=maximum)
