import math
import numpy as np
import torch
from .common import G,RN,OUT,write_json,now
from .resources import wait_headroom

def log_joint(n,lam,k,b,g,rn,event):
    lp=-lam+n*torch.log(lam.clamp_min(1e-300))-torch.lgamma(n+1)
    lp=torch.where((lam==0)&(n>0),-torch.inf,lp)
    mu=b+g*n
    if event=='bin':
        lo=(k-.5-mu)/rn; hi=(k+.5-mu)/rn
        a=torch.special.log_ndtr(torch.where(lo>=0,-lo,hi))
        z=torch.special.log_ndtr(torch.where(lo>=0,-hi,lo))
        value=a+torch.log(-torch.expm1(z-a))
    elif event=='upper': value=torch.special.log_ndtr((mu-k)/rn)
    else: value=torch.special.log_ndtr((k-mu)/rn)
    return lp+value

def adaptive(lam,k,b,g,rn,event='bin',device='cpu',radius=96):
    lam,k=torch.broadcast_tensors(torch.as_tensor(lam,dtype=torch.float64,device=device),torch.as_tensor(k,dtype=torch.float64,device=device))
    shape=lam.shape; lam=lam.reshape(-1,1); k=k.reshape(-1,1)
    sig2=rn**2+1/12
    mode=((k-b)/g).clamp_min(.5)
    if event=='upper':


        left=torch.zeros_like(lam)
        right=(torch.maximum(lam,mode)+32*rn/g+512).ceil()
        for _ in range(32):
            mid=((left+right)/2).floor()
            ratio=log_joint(mid+1,lam,k,b,g,rn,event)-log_joint(mid,lam,k,b,g,rn,event)
            ratio=torch.where(lam==0,-torch.inf,ratio)
            moving=left<right
            left=torch.where(moving&(ratio>=0),mid+1,left)
            right=torch.where(moving&(ratio<0),mid,right)
        assert bool((left==right).all())
        mode=left; radius=max(radius,int(math.ceil(12*math.sqrt(float(lam.max()))))+128)
    else:
        for _ in range(12):
            derivative=torch.log(lam.clamp_min(1e-300))-torch.log(mode+.5)+g*(k-b-g*mode)/sig2
            mode=(mode+derivative/(1/(mode+.5)+g*g/sig2)).clamp_min(0)
    mode=torch.where(lam==0,0,mode).round()
    lower=(mode-radius).clamp_min(0); upper=mode+radius

    n=lower+torch.arange(2*radius+1,device=device,dtype=torch.float64)[None]
    upper=n[:,-1:]; terms=log_joint(n,lam,k,b,g,rn,event)
    total=torch.logsumexp(terms,1,keepdim=True)
    next_right=log_joint(upper+1,lam,k,b,g,rn,event); right_ratio=next_right-terms[:,-1:]
    right_ratio=torch.where(torch.isneginf(next_right),-torch.inf,right_ratio)
    assert bool((right_ratio<0).all()),'Adaptive mode window missed right mode'
    right_bound=next_right-torch.log(-torch.expm1(right_ratio))
    left_n=(lower-1).clamp_min(0); left=log_joint(left_n,lam,k,b,g,rn,event)
    left_ratio=left-terms[:,:1]
    left_ratio=torch.where(lower==0,-torch.inf,left_ratio)
    assert bool((left_ratio<0).all()),'Adaptive mode window missed left mode'
    left_bound=torch.where(lower==0,-torch.inf,left-torch.log(-torch.expm1(left_ratio)))
    relative=torch.logaddexp(left_bound,right_bound)-total
    maximum=float(relative.max())
    assert maximum<math.log(1e-13),(maximum,float(lam.max()),float(k.max()))
    return total.reshape(shape),maximum

@torch.no_grad()
def pg_table(grid,b,g,rn,max_y,observed_codes=None,device=None):
    wait_headroom(2)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    positives=np.arange(1,max_y+1) if observed_codes is None else np.unique(np.asarray(observed_codes,dtype=int))
    assert max_y<=16383,'This likelihood table is for the fixed 14bit ADC'
    upper_present=bool(np.any(positives==16383))
    positives=positives[(positives>0)&(positives<16383)]; codes=np.r_[np.arange(-64,1),positives]
    logmass=np.empty((len(grid),len(codes))); maximum_bound=-math.inf
    for start in range(0,len(grid),128):
        gl=torch.as_tensor(grid[start:start+128],dtype=torch.float64,device=device)[:,None]
        for j in range(0,len(codes),16):
            ks=torch.as_tensor(codes[j:j+16],dtype=torch.float64,device=device)[None,:]
            values,bound=adaptive(gl,ks,b,g,rn,device=device)
            logmass[start:start+len(gl),j:j+ks.shape[1]]=values.cpu().numpy(); maximum_bound=max(maximum_bound,bound)
    loga=np.empty(len(grid)); logtail=np.empty(len(grid))
    for start in range(0,len(grid),1024):
        gl=torch.as_tensor(grid[start:start+1024],dtype=torch.float64,device=device)
        a,ba=adaptive(gl,.5,b,g,rn,'cdf',device); t,bt=adaptive(gl,-64.5,b,g,rn,'cdf',device)
        loga[start:start+len(gl)]=a.cpu().numpy(); logtail[start:start+len(gl)]=(t-a).cpu().numpy()
        maximum_bound=max(maximum_bound,ba,bt)
    from scipy.special import logsumexp
    lq=logmass[:,:65]-loga[:,None]; error=float(np.max(abs(np.exp(lq).sum(1)-1)))
    assert error<1e-9,error
    lq-=logsumexp(lq,axis=1)[:,None]
    ll=np.full((len(grid),max_y+1),-np.inf); ll[:,0]=loga; ll[:,positives]=logmass[:,65:]
    if upper_present:
        for start in range(0,len(grid),32):
            gl=torch.as_tensor(grid[start:start+32],dtype=torch.float64,device=device)
            upper,bound=adaptive(gl,16382.5,b,g,rn,'upper',device)
            ll[start:start+len(gl),16383]=upper.cpu().numpy(); maximum_bound=max(maximum_bound,bound)
    return dict(ll=ll,lq=lq,loga=loga,logtail=logtail,normalization_error=error,max_log_relative_omitted_mass=maximum_bound)

def audit():
    from physics.pg_quantized import pg_logpmf,pg_zero_event_logprob,poisson_n_max,_log_pois,normal_interval_logprob
    from scipy.special import logsumexp,log_ndtr
    torch.set_num_threads(2); errors=[]; bounds=[]; full_errors=[]
    for b in [-.25,.25]:
        for lam in [0,.001,.1,1,10,100,1000,8000]:
            for k in [-64,-5,0,1,50,500,1500]:
                out,bound=adaptive(lam,k,b,G,RN,device='cuda')
                ref=float(pg_logpmf(k,lam,b,G,RN,tol=1e-13)); errors.append(abs(float(out)-ref)); bounds.append(bound)


                end=max(poisson_n_max(lam,tol=1e-13),int(max(k,0)/G+32*RN/G)+512)
                ns=np.arange(end+1); full=logsumexp(_log_pois(ns,lam)+normal_interval_logprob(k-.5-b-G*ns,k+.5-b-G*ns,RN))
                full_errors.append(abs(float(out)-full))
            for k in [.5,-64.5]:
                out,bound=adaptive(lam,k,b,G,RN,'cdf',device='cuda')
                n=np.arange(poisson_n_max(lam,tol=1e-13)+1)
                ref=logsumexp(_log_pois(n,lam)+log_ndtr((k-b-G*n)/RN))
                errors.append(abs(float(out)-ref)); bounds.append(bound)



    near=[]
    for lam in [0,.001,1,100,1000,8000]:
        for k in [0,-1,-64,int(round(G*lam))]:
            out,bound=adaptive(lam,k,0,G,RN,device='cuda')
            ref=float(pg_logpmf(k,lam,0,G,RN,tol=1e-13)); near.append(abs(float(out)-ref))
    report=dict(status='PASS' if max(near)<1e-7 and max(full_errors)<1e-7 else 'FAIL',time=now(),max_near_and_negative_log_error=max(near),
        max_independent_observation_extended_reference_log_error=max(full_errors),
        old_absolute_prior_truncation_extreme_event_max_difference=max(errors),max_log_relative_omitted_mass=max(bounds),
        algorithm='integer Poisson-Gaussian sum around mode; geometric posterior tail bound per query; exact interval probabilities',
        sparse_observed_codes=True)
    write_json(OUT/'ADAPTIVE_PG_AUDIT.json',report); print(report,flush=True); assert report['status']=='PASS'

if __name__=='__main__': audit()
