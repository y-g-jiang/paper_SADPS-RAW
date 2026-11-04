import math
import numpy as np
from scipy.special import gammaln,log_ndtr,logsumexp
from .pg_quantized import normal_interval_logprob

def joint(n,lam,k,b,g,rn,event):
    with np.errstate(divide='ignore',invalid='ignore'):
        lp=-lam+n*np.log(np.maximum(lam,1e-300))-gammaln(n+1)
    lp=np.where((lam==0)&(n>0),-np.inf,lp); mu=b+g*n
    if event=='bin': likelihood=normal_interval_logprob(k-.5-mu,k+.5-mu,rn)
    else: likelihood=log_ndtr((mu-k)/rn if event=='upper' else (k-mu)/rn)
    return lp+likelihood

def adaptive(lam,k,b,g,rn,event='bin',radius=96):
    lam,k=np.broadcast_arrays(np.asarray(lam,np.float64),np.asarray(k,np.float64)); shape=lam.shape
    lam=lam.reshape(-1,1); k=k.reshape(-1,1); mode=np.maximum((k-b)/g,.5)
    if event=='upper':
        left=np.zeros_like(lam); right=np.ceil(np.maximum(lam,mode)+32*rn/g+512)
        for _ in range(32):
            mid=np.floor((left+right)/2)
            with np.errstate(invalid='ignore'): ratio=joint(mid+1,lam,k,b,g,rn,event)-joint(mid,lam,k,b,g,rn,event)
            ratio=np.where(lam==0,-np.inf,ratio); moving=left<right
            left=np.where(moving&(ratio>=0),mid+1,left); right=np.where(moving&(ratio<0),mid,right)
        assert np.all(left==right)
        mode=left; radius=max(radius,int(math.ceil(12*np.sqrt(lam.max())))+128)
    else:
        variance=rn**2+1/12
        for _ in range(12):
            derivative=np.log(np.maximum(lam,1e-300))-np.log(mode+.5)+g*(k-b-g*mode)/variance
            mode=np.maximum(mode+derivative/(1/(mode+.5)+g*g/variance),0)
    mode=np.where(lam==0,0,np.rint(mode)); lower=np.maximum(mode-radius,0)
    ns=lower+np.arange(2*radius+1)[None]; upper=ns[:,-1:]
    terms=joint(ns,lam,k,b,g,rn,event); total=logsumexp(terms,axis=1,keepdims=True)
    right=joint(upper+1,lam,k,b,g,rn,event); left=joint(np.maximum(lower-1,0),lam,k,b,g,rn,event)
    with np.errstate(invalid='ignore',divide='ignore'):
        rr=np.where(np.isneginf(right),-np.inf,right-terms[:,-1:])
        lr=np.where(lower==0,-np.inf,left-terms[:,:1])
        assert np.all(rr<0) and np.all(lr<0)
        rb=right-np.log(-np.expm1(rr)); lb=np.where(lower==0,-np.inf,left-np.log(-np.expm1(lr)))
    maximum=float(np.max(np.logaddexp(lb,rb)-total)); assert maximum<math.log(1e-13),maximum
    return total.reshape(shape),maximum

def pg_table(grid,b,g,rn,max_y,observed_codes=None,device=None):
    assert max_y<=16383
    positives=np.arange(1,max_y+1) if observed_codes is None else np.unique(np.asarray(observed_codes,dtype=int))
    upper_present=bool(np.any(positives==16383)); positives=positives[(positives>0)&(positives<16383)]
    codes=np.r_[np.arange(-64,1),positives]; logmass=np.empty((len(grid),len(codes))); bound=-math.inf
    for start in range(0,len(grid),16):
        gl=grid[start:start+16,None]
        for j in range(0,len(codes),8):
            ks=codes[None,j:j+8]; value,bv=adaptive(gl,ks,b,g,rn)
            logmass[start:start+len(gl),j:j+ks.shape[1]]=value; bound=max(bound,bv)
    loga,ba=adaptive(grid,.5,b,g,rn,'cdf'); tail,bt=adaptive(grid,-64.5,b,g,rn,'cdf')
    lq=logmass[:,:65]-loga[:,None]; error=float(abs(np.exp(lq).sum(1)-1).max()); assert error<1e-9
    lq-=logsumexp(lq,axis=1)[:,None]; ll=np.full((len(grid),max_y+1),-np.inf); ll[:,0]=loga; ll[:,positives]=logmass[:,65:]
    if upper_present:
        for start in range(0,len(grid),16):
            value,bv=adaptive(grid[start:start+16],16382.5,b,g,rn,'upper'); ll[start:start+len(value),16383]=value; bound=max(bound,bv)
    return dict(ll=ll,lq=lq,loga=loga,logtail=tail-loga,normalization_error=error,max_log_relative_omitted_mass=max(bound,ba,bt))
