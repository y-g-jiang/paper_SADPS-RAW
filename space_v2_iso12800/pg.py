import math
import numpy as np
import torch
from scipy.special import log_ndtr, logsumexp
from physics.pg_quantized import pg_logpmf, pg_zero_event_logprob, poisson_n_max, _log_pois
from .common import *

def reference(s,b):
    lam=float(s)/G; n=np.arange(poisson_n_max(lam,tol=1e-13)+1)
    a=float(pg_zero_event_logprob(lam,b,G,RN,tol=1e-13))
    lp=pg_logpmf(-np.arange(K+1),lam,b,G,RN,tol=1e-13)-a
    tail=logsumexp(_log_pois(n,lam)+log_ndtr(((-K-.5)-b-G*n)/RN))-a
    return np.r_[lp,tail]

def exact(s,b):

    s,b=torch.broadcast_tensors(s.double(),b.double()); shape=s.shape
    s=s.reshape(-1); b=b.reshape(-1); lam=s/G
    nmax=poisson_n_max(float(lam.detach().max()),tol=1e-13)
    n=torch.arange(nmax+1,dtype=torch.float64,device=s.device)
    lp=-lam[:,None]+n*torch.log(lam[:,None].clamp_min(1e-300))-torch.lgamma(n+1)
    lp=torch.where((lam[:,None]==0)&(n>0),-torch.inf,lp)
    mu=b[:,None]+G*n
    loga=torch.logsumexp(lp+torch.special.log_ndtr((.5-mu)/RN),-1)
    codes=-torch.arange(K+1,dtype=torch.float64,device=s.device)
    hi=torch.special.log_ndtr((codes[None,:,None]+.5-mu[:,None,:])/RN)
    lo=torch.special.log_ndtr((codes[None,:,None]-.5-mu[:,None,:])/RN)
    interval=hi+torch.log(-torch.expm1(lo-hi))
    lq=torch.logsumexp(lp[:,None,:]+interval,-1)-loga[:,None]
    tail=torch.logsumexp(lp+torch.special.log_ndtr(((-K-.5)-mu)/RN),-1)-loga
    return torch.cat((lq,tail[:,None]),1).reshape(*shape,K+2)

class Decoder(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        p=np.load(OUT/'PG_LUT.npz')
        self.dx=float(p['dx']); self.db=float(p['db']); self.xmax=float(p['xmax'])
        self.register_buffer('table',torch.from_numpy(p['logq']).to(device))

    def forward(self,s,b):
        s,b=torch.broadcast_tensors(s.float(),b.float())
        assert bool(torch.isfinite(s).all()) and bool((s>=0).all())
        assert bool((b.abs()<=.25001).all())
        x=torch.log1p(s)/self.dx; z=(b+.25)/self.db
        ix=x.long().clamp(0,self.table.shape[1]-2); iz=z.long().clamp(0,self.table.shape[0]-2)
        ax=(x-ix).unsqueeze(-1); az=(z-iz).unsqueeze(-1)
        q=(1-az)*((1-ax)*self.table[iz,ix]+ax*self.table[iz,ix+1])+az*((1-ax)*self.table[iz+1,ix]+ax*self.table[iz+1,ix+1])
        outside=x>self.table.shape[1]-1
        if bool(outside.any()): q=q.index_put((outside,),exact(s[outside],b[outside]).float())
        return q-torch.logsumexp(q,-1,keepdim=True)

def build():
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    dx=.002; db=.01; xs=np.arange(math.ceil(math.log(65)/dx)+1)*dx
    bs=np.linspace(-.25,.25,51); table=np.empty((len(bs),len(xs),K+2),np.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for j,b in enumerate(bs):
        for start in range(0,len(xs),128):
            s=torch.tensor(np.expm1(xs[start:start+128]),device=device)
            with torch.no_grad(): table[j,start:start+len(s)]=exact(s,torch.full_like(s,b)).cpu().numpy()
        if j%5==0: print('PG table',j,len(bs),flush=True)
    np.savez(OUT/'PG_LUT.npz',logq=table,dx=dx,db=db,xmax=xs[-1])
    d=Decoder(device); rng=np.random.default_rng(20261025)
    values=[(float(np.expm1(rng.uniform(0,np.log(65)))),float(rng.uniform(-.25,.25))) for _ in range(128)]
    values += [(s,b) for s in [0,.01,.1,1,5,16,64,80] for b in [-.25,0,.25]]
    errors=[]; kls=[]; dense_errors=[]; norm=[]; tails=[]
    for s,b in values:
        ref=reference(s,b)
        with torch.no_grad(): approx=d(torch.tensor([s],device=device),torch.tensor([b],device=device))[0].double().cpu().numpy()
        p=np.exp(ref); errors.append(abs(approx-ref)); kls.append(float(np.sum(p*(ref-approx))))
        norm.append(abs(p.sum()-1)); tails.append(float(ref[-1]))

        x=np.log1p(s); ix=np.floor(x/(dx/2)); iz=np.floor((b+.25)/(db/2))
        ax=x/(dx/2)-ix; az=(b+.25)/(db/2)-iz
        fine=sum(w*reference(np.expm1((ix+a)*dx/2),-.25+(iz+c)*db/2)
                 for a,c,w in [(0,0,(1-ax)*(1-az)),(1,0,ax*(1-az)),(0,1,(1-ax)*az),(1,1,ax*az)])
        fine-=logsumexp(fine); dense_errors.append(float(np.sum(p*np.abs(approx-fine))))
    audit=dict(status='PASS' if max(kls)<1e-5 and max(dense_errors)<1e-4 else 'FAIL',
        comparisons=len(values),mean_KL=float(np.mean(kls)),max_KL=max(kls),max_abs_logq=float(np.max(errors)),
        mean_abs_logq=float(np.mean(errors)),max_mass_error=max(norm),
        densification_max_teacher_weighted_abs_logq=max(dense_errors),max_log_tail=max(tails),
        support=f'coarsened 0..-{K} plus genuine tail; never an atom',
        dx=dx,db=db,table_sha256=sha(OUT/'PG_LUT.npz'),outside_policy='exact FP64 PG with autograd')
    write_json(OUT/'PMF_NUMERIC_AUDIT.json',audit); print(audit,flush=True)
    assert audit['status']=='PASS'

if __name__=='__main__': build()
