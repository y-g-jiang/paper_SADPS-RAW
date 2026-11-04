import numpy as np
import torch
from scipy.special import logsumexp,log_ndtr
from .common import G,RN,K
from .pg_quantized import poisson_n_max
from .cpu_pg import adaptive

@torch.inference_mode()
def exact(s,b):
    s,b=torch.broadcast_tensors(s.double(),b.double());shape=s.shape
    s=s.reshape(-1);b=b.reshape(-1);lam=s/G
    n=torch.arange(poisson_n_max(float(lam.max()),tol=1e-13)+1,dtype=torch.float64,device=s.device)
    lp=-lam[:,None]+n*torch.log(lam[:,None].clamp_min(1e-300))-torch.lgamma(n+1)
    lp=torch.where((lam[:,None]==0)&(n>0),-torch.inf,lp);mu=b[:,None]+G*n
    den=torch.logsumexp(lp+torch.special.log_ndtr((.5-mu)/RN),-1)
    codes=-torch.arange(K+1,dtype=torch.float64,device=s.device)
    hi=torch.special.log_ndtr((codes[None,:,None]+.5-mu[:,None,:])/RN)
    lo=torch.special.log_ndtr((codes[None,:,None]-.5-mu[:,None,:])/RN)
    interval=hi+torch.log(-torch.expm1(lo-hi))
    q=torch.logsumexp(lp[:,None,:]+interval,-1)-den[:,None]
    tail=torch.logsumexp(lp+torch.special.log_ndtr(((-K-.5)-mu)/RN),-1)-den
    return torch.cat((q,tail[:,None]),1).reshape(*shape,K+2)

def score(q,signals,b,v):
    q=np.asarray(q,float);q=q-logsumexp(q,axis=1)[:,None];v=np.asarray(v)
    b=np.broadcast_to(np.asarray(b,float),v.shape);signals=np.asarray(signals,float)
    assert np.isfinite(q).all() and (v<=0).all() and (v==np.floor(v)).all()
    Q=np.exp(q);w=-v.astype(int);cdf=Q.cumsum(1)
    nll=-q[np.arange(len(v)),np.minimum(w,K+1)]
    crps=((cdf[:,:K+1]-(np.arange(K+1)[None]>=w[:,None]))**2).sum(1)+np.maximum(w-K-1,0)
    for i in np.flatnonzero(w>K):
        den=float(adaptive(signals[i]/G,-K-.5,b[i],G,RN,'cdf')[0])
        mass=float(adaptive(signals[i]/G,int(v[i]),b[i],G,RN,'bin')[0])
        nll[i]=-(q[i,-1]+mass-den)
    lr=log_ndtr((-K-1.5-b)/RN)-log_ndtr((-K-.5-b)/RN);d=-np.expm1(lr)
    bound=Q[:,-1]*np.exp(lr)*((2*K+3)/d+2*np.exp(lr)/d**2)
    assert bound.max()<1e-12
    return {'n_candidates':len(v),'NLL':float(nll.mean()),'CRPS':float(crps.mean()),'CRPS_integer_tail_abs_bound':float(bound.max()),'observed_negative_fraction':float((v<0).mean()),'predicted_negative_fraction':float((1-Q[:,0]).mean())}
