import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson
from baselines.foi_adapter import FoiTransform,bm3d
from physics.pg_quantized import normal_interval_logprob
from .common import G,RN

def quantized_inverse(transform,filtered,max_observed):
    maximum=max(32.,float(max_observed)+16)
    signal=np.unique(np.r_[np.arange(0,16,.02),np.arange(16,min(64,maximum),.1),
                           np.arange(64,maximum,.5) if maximum>64 else [],maximum])
    lam=signal/G;expected=np.empty_like(lam)
    radius=int(np.ceil(12*RN))+2; offsets=np.arange(-radius,radius+1)
    bound=0.
    for start in range(0,len(lam),32):
        ll=lam[start:start+32]
        low=max(0,int(poisson.ppf(1e-13,ll[0])))
        high=int(poisson.ppf(1-1e-13,ll[-1]))+2
        n=np.arange(low,high+1);conditional=np.empty(len(n))
        for a in range(0,len(n),256):
            mu=transform.b+G*n[a:a+256,None]
            code=np.floor(mu+.5)+offsets
            mass=np.exp(normal_interval_logprob(code-.5-mu,code+.5-mu,RN))
            observation=np.clip(code,0,16383)
            conditional[a:a+256]=(mass*transform.forward(observation)).sum(1)
        logw=-ll[:,None]+n*np.log(np.maximum(ll[:,None],1e-300))-gammaln(n+1)
        if ll[0]==0:logw[0,n>0]=-np.inf
        expected[start:start+32]=np.exp(logw)@conditional
        omitted=poisson.cdf(low-1,ll)+poisson.sf(high,ll)
        bound=max(bound,float(omitted.max()))
    assert np.all(np.diff(expected)>0),'Quantized inverse must be strictly monotone'
    return np.interp(filtered,expected,lam),dict(grid_points=len(lam),upper_projection_count=int((filtered>expected[-1]).sum()),
        max_poisson_omitted_probability=bound,gaussian_tail_beyond_12sigma_omitted=True)

def infer(y,valid,b):
    yy,xx=np.nonzero(valid);top,bottom=yy.min(),yy.max()+1;left,right=xx.min(),xx.max()+1
    assert valid[top:bottom,left:right].all() and int(valid.sum())==(bottom-top)*(right-left)
    observed=np.asarray(y[top:bottom,left:right],np.float64)
    maximum=max(256.,float(observed.max())+36+12*np.sqrt(G*(float(observed.max())+16)+RN**2))
    transform=FoiTransform(G,RN,b,maximum=maximum)
    transformed=transform.forward(observed)
    profile=bm3d.BM3DProfile();profile.num_threads=1
    filtered=bm3d.bm3d(transformed,sigma_psd=1.,profile=profile)
    lam,meta=quantized_inverse(transform,filtered,observed.max())
    signal=np.zeros_like(y,dtype=np.float32);signal[top:bottom,left:right]=(lam*G).astype(np.float32)
    return signal,dict(**meta,valid_shape=list(observed.shape),input_max=float(observed.max()),backend_threads=1)
