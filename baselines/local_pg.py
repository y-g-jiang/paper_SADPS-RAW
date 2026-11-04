import time
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma
from physics.pg_quantized import normal_interval_logprob, poisson_n_max
from scipy.special import log_ndtr

def box_sum(x,width):
    return uniform_filter(np.asarray(x,dtype=np.float64),size=(1,width,width) if x.ndim==3 else width,mode='constant')*width**2

def lambda_grid(max_signal,g,step_e=.5):

    end=max(64.,max_signal/g)
    return np.unique(np.r_[np.arange(0,min(end,128)+step_e*.5,step_e),
        np.arange(128+step_e*2,end+step_e*2,step_e*4) if end>128 else [],end])

def pg_table(grid,b,g,rn,max_y):

    n=np.arange(poisson_n_max(float(grid.max()),tol=1e-13)+1,dtype=float)
    lp=-grid[:,None]+n[None,:]*np.log(np.maximum(grid[:,None],1e-300))-gammaln(n+1)
    lp[(grid==0),1:]=-np.inf
    mu=b+g*n
    codes=np.arange(-64,max_y+1)
    logmass=np.empty((len(grid),len(codes)))
    for j,k in enumerate(codes):
        interval=normal_interval_logprob(k-.5-mu,k+.5-mu,rn)
        logmass[:,j]=logsumexp(lp+interval[None,:],axis=1)
    loga=logsumexp(lp+log_ndtr((.5-mu)/rn),axis=1)
    logtail=logsumexp(lp+log_ndtr((-64.5-mu)/rn),axis=1)-loga
    lq=logmass[:,:65]-loga[:,None]

    error=float(np.max(abs(np.exp(lq).sum(1)-1)))
    assert error<1e-10
    lq-=logsumexp(lq,axis=1)[:,None]
    ll=logmass[:,64:].copy();ll[:,0]=loga
    return {'ll':ll,'lq':lq,'loga':loga,'logtail':logtail,'normalization_error':error}

def prior_weights(grid,prior):
    p0=prior['zero_mass'];shape=prior['gamma_shape'];scale=prior['gamma_scale_e']
    pos=grid[1:];edges=np.r_[0,(pos[:-1]+pos[1:])/2,pos[-1]+(pos[-1]-pos[-2])/2]
    weights=np.r_[p0,(1-p0)*np.diff(gamma.cdf(edges,shape,scale=scale))]
    return np.maximum(weights,1e-300)

def bayes_combine(neighbor_loglike,table,prior_mass):

    lr=neighbor_loglike+np.log(prior_mass)[None,:]
    plus=lr+table['loga'][None,:]
    plus-=logsumexp(plus,axis=1)[:,None]
    weights=np.exp(plus)
    q=weights@np.exp(table['lq'])
    q/=q.sum(1)[:,None]
    return q,weights

def estimate_channel(y,b,g,rn,prior,widths=(5,9,15),step_e=.5):
    mask=y==0;n=int(mask.sum())
    if not n:return {},{'n':0}
    count={w:box_sum(np.ones_like(y),w) for w in widths}
    means={w:box_sum(y,w)/count[w] for w in widths}
    max_local=max(float(means[w][mask].max()) for w in widths)
    end_signal=max(8.,1.8*max_local+6.)
    grid=lambda_grid(end_signal,g,step_e)


    start=time.perf_counter();table=pg_table(grid,b,g,rn,int(y.max()))
    llmap=table['ll'][:,y.astype(int)]
    outputs={};stats={'n':n,'lambda_grid_max_e':float(grid[-1]),'grid_points':len(grid),
        'lambda_step_low_e':step_e,'table_seconds':time.perf_counter()-start,
        'table_mass_error':table['normalization_error'],'max_log_tail':float(table['logtail'].max()),'windows':{}}
    for w in widths:
        local=box_sum(llmap,w)[:,mask].T
        mle_idx=np.argmax(local,axis=1)
        lam=grid[mle_idx].copy()
        interior=(mle_idx>0)&(mle_idx<len(grid)-1)
        rows=np.flatnonzero(interior);ii=mle_idx[interior]
        hm=grid[ii]-grid[ii-1];hp=grid[ii+1]-grid[ii]
        fm,f0,fp=local[rows,ii-1],local[rows,ii],local[rows,ii+1]
        dm=(f0-fm)/hm;dp=(fp-f0)/hp;aa=(dp-dm)/(hm+hp);bb=dm+aa*hm
        shift=np.where(aa<0,-bb/(2*np.minimum(aa,-1e-30)),0)
        lam[interior]+=np.clip(shift,-hm,hp)
        mi=np.clip(np.searchsorted(grid,lam)-1,0,len(grid)-2)
        ma=((lam-grid[mi])/(grid[mi+1]-grid[mi]))[:,None]
        lq=(1-ma)*table['lq'][mi]+ma*table['lq'][mi+1]
        lq-=logsumexp(lq,axis=1)[:,None];q=np.exp(lq)
        neighbors=local-table['loga'][None,:]
        qb,weights=bayes_combine(neighbors,table,prior_weights(grid,prior))
        lb_lam=weights@grid


        idx=np.clip(np.searchsorted(grid,lb_lam)-1,0,len(grid)-2)
        alpha=((lb_lam-grid[idx])/(grid[idx+1]-grid[idx]))[:,None]
        lqp=table['lq'][idx]*(1-alpha)+table['lq'][idx+1]*alpha
        lqp-=logsumexp(lqp,axis=1)[:,None]
        outputs[w]={'L':(q,lq,lam),'LB':(qb,np.log(np.maximum(qb,1e-300)),lb_lam),
                    'LB-plugin':(np.exp(lqp),lqp,lb_lam)}
        stats['windows'][str(w)]={'mle_zero_boundary':int((mle_idx==0).sum()),
            'mle_upper_boundary':int((mle_idx==len(grid)-1).sum()),
            'bayes_last_two_grid_mass_max':float(weights[:,-2:].sum(1).max())}
        assert not (mle_idx==len(grid)-1).any(), 'Enlarge local grid'
        assert weights[:,-2:].sum(1).max()<1e-5, 'Enlarge Bayesian grid'
    return outputs,stats

def local_distributions(y,known,prior,widths=(5,9,15),step_e=.5):
    chunks={w:{k:[] for k in ('L','LB','LB-plugin')} for w in widths};stats=[]
    for c in range(4):
        out,s=estimate_channel(y[c],known['b'][c],known['gain'][c],known['read_noise'][c],prior,widths,step_e)
        stats.append(s)
        if not s['n']:continue
        for w in widths:
            for k in chunks[w]:chunks[w][k].append(out[w][k])
    return {w:{k:tuple(np.concatenate([v[j] for v in parts]) for j in range(3)) for k,parts in methods.items()} for w,methods in chunks.items()},stats
