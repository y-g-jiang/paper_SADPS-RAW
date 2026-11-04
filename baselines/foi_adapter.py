import sys,time
from functools import lru_cache
import numpy as np
from scipy.special import ndtr
from scipy.integrate import cumulative_trapezoid
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln
from physics.pg_quantized import poisson_n_max,normal_interval_logprob

import bm3d

class FoiTransform:
    def __init__(self,g,rn,b,maximum=256.,resolution=1,quadrature=192):
        self.g,self.rn,self.b=float(g),float(rn),float(b)
        lower=b-rn*rn/g
        mu=np.unique(np.r_[np.linspace(lower+1e-4,-4,600*resolution),
                          np.arange(-4,16,.005/resolution),np.arange(16,maximum+.05,.05/resolution)])
        sd=np.sqrt(np.maximum(rn*rn+g*(mu-b),1e-12))
        t=mu/sd;phi=np.exp(-t*t/2)/np.sqrt(2*np.pi);cdf=ndtr(t)
        clipped_mean=mu*cdf+sd*phi
        second=(mu*mu+sd*sd)*cdf+mu*sd*phi
        clipped_sd=np.sqrt(np.maximum(second-clipped_mean**2,1e-300))
        good=(clipped_mean>1e-15)&(clipped_sd>1e-12)
        mu,sd,cm,cs=mu[good],sd[good],clipped_mean[good],clipped_sd[good]

        f=cumulative_trapezoid(1/cs,cm,initial=0)
        self.x=np.r_[0,cm];self.f=np.r_[0,f];self.mu=mu;self.cm=cm
        self.fm=np.interp(cm,self.x,self.f)
        nodes,weights=leggauss(quadrature)
        ef=np.empty_like(mu)
        for st in range(0,len(mu),512):
            mm=mu[st:st+512,None];ss=sd[st:st+512,None]

            low=np.maximum(-mm/ss,-12);high=np.full_like(low,12)
            zz=(nodes[None,:]+1)/2*(high-low)+low
            values=np.maximum(mm+ss*zz,0)
            ff=np.interp(values,self.x,self.f)
            ef[st:st+512]=((ff*np.exp(-zz*zz/2)/np.sqrt(2*np.pi))*weights[None,:]*(high-low)/2).sum(1)
        keep=np.r_[True,np.diff(ef)>1e-12]
        self.ef=ef[keep];self.h_inverse=self.fm[keep]
        self.meta={'mu_grid':len(mu),'quadrature':quadrature,'resolution':resolution,'f_at_zero':0.,
                   'expectation_monotone':bool(np.all(np.diff(self.ef)>0)),'removed_flat_knots':int((~keep).sum())}
        assert np.isfinite(ef).all() and self.meta['expectation_monotone']

    def forward(self,y):
        if float(y.max())>self.x[-1]:raise ValueError('Increase transform range')
        return np.interp(y,self.x,self.f)

    def inverse_steps(self,denoised):
        projected=np.clip(denoised,self.ef[0],self.ef[-1])
        f_mean=np.interp(projected,self.ef,self.h_inverse)
        mean_clipped=np.interp(f_mean,self.f,self.x)
        mean_preclip=np.interp(mean_clipped,self.cm,self.mu)
        return mean_preclip

    def quantized_inverse(self,denoised,max_observed):


        maximum=max(32.,float(max_observed)+16)
        signal=np.unique(np.r_[np.arange(0,16,.02),np.arange(16,min(64,maximum),.1),
                               np.arange(64,maximum,.5) if maximum>64 else [],maximum])
        lam=signal/self.g;n=np.arange(poisson_n_max(lam[-1],tol=1e-13)+1)
        codes=np.arange(1,int(np.ceil(self.b+self.g*n[-1]+12*self.rn))+1)
        mu=self.b+self.g*n[:,None]
        mass=np.exp(normal_interval_logprob(codes[None,:]-.5-mu,codes[None,:]+.5-mu,self.rn))
        conditional=mass@self.forward(codes)
        lp=-lam[:,None]+n*np.log(np.maximum(lam[:,None],1e-300))-gammaln(n+1)
        lp[0,1:]=-np.inf;expectation=np.exp(lp)@conditional
        assert np.all(np.diff(expectation)>0), 'Non-monotone quantized expectation LUT'
        result=np.interp(denoised,expectation,lam)
        return result,{'grid_points':len(lam),'lambda_max_e':float(lam[-1]),'expectation_monotone':True,
            'upper_projection_count':int((denoised>expectation[-1]).sum())}

def foi_signal(y,known,include_quantized=False):
    result=np.empty_like(y,dtype=float);adapted=np.empty_like(result);timing={'transform_build_seconds':0.,'filter_seconds':0.,'inverse_seconds':0.,'quantized_mapping_seconds':0.};meta=[]
    for c in range(4):
        g,rn,b=[float(known[k][c]) for k in ('gain','read_noise','b')]
        t=time.perf_counter();transform=FoiTransform(g,rn,b,maximum=max(256,float(y[c].max())+128));timing['transform_build_seconds']+=time.perf_counter()-t
        transformed=transform.forward(y[c]);t=time.perf_counter()
        profile=bm3d.BM3DProfile();profile.num_threads=4
        filtered=bm3d.bm3d(transformed,sigma_psd=1.,profile=profile)
        timing['filter_seconds']+=time.perf_counter()-t;t=time.perf_counter()
        mu=transform.inverse_steps(filtered);result[c]=np.maximum((mu-b)/g,0)
        timing['inverse_seconds']+=time.perf_counter()-t
        if include_quantized:
            t=time.perf_counter();adapted[c],qm=transform.quantized_inverse(filtered,y[c].max());timing['quantized_mapping_seconds']+=time.perf_counter()-t;transform.meta['quantized']=qm
        meta.append(transform.meta)
    return (result,adapted,timing,meta) if include_quantized else (result,timing,meta)
