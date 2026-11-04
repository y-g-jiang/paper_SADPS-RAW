import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import logsumexp
from baselines.local_pg import lambda_grid,prior_weights

def pg_table(*args,device=None,**kwargs):
    if device=='cpu':
        from .cpu_pg import pg_table as implementation
    else:
        from .fast_pg import pg_table as implementation
    return implementation(*args,device=device,**kwargs)

def histogram_at(y,valid,centers,width,codes):
    result=np.empty((len(centers),len(codes)),np.float64)
    for j,code in enumerate(codes):
        counts=uniform_filter(((y==code)&valid).astype(np.float64),size=width,mode='constant')*width**2
        result[:,j]=counts[centers[:,0],centers[:,1]]
    return result

def refined_mle(grid,likelihood):
    indices=np.argmax(likelihood,axis=1); estimates=grid[indices].copy()
    assert not np.any(indices==len(grid)-1),'Enlarge likelihood parameter grid'
    inside=(indices>0)&(indices<len(grid)-1); rows=np.flatnonzero(inside); ii=indices[inside]
    hm=grid[ii]-grid[ii-1]; hp=grid[ii+1]-grid[ii]
    fm=likelihood[rows,ii-1]; f0=likelihood[rows,ii]; fp=likelihood[rows,ii+1]
    dm=(f0-fm)/hm; dp=(fp-f0)/hp; a=(dp-dm)/(hm+hp); b=dm+a*hm
    shift=np.where(a<0,-b/(2*np.minimum(a,-1e-30)),0)
    estimates[inside]+=np.clip(shift,-hm,hp)
    return estimates

def infer_channel(y,valid,mask,b,g,rn,prior,topks=(4,8),reference_stride=8,device=None):
    assert reference_stride>=5,'Reference windows must not reuse overlapping observations'
    assert np.isfinite(y).all() and np.all((y>=0)&(y<=16383)&(y==np.floor(y)))
    y=np.asarray(y,dtype=np.int64); valid=np.asarray(valid,dtype=bool); mask=np.asarray(mask,dtype=bool)&valid&(y==0)
    centers=np.argwhere(mask)
    if not len(centers): return {},dict(candidates=0)

    maximum=0.
    for width in [5,9,15]:
        mean=uniform_filter(y.astype(float)*valid,width,mode='constant')/np.maximum(uniform_filter(valid.astype(float),width,mode='constant'),1e-12)
        maximum=max(maximum,float(mean[mask].max()))
    grid=lambda_grid(max(8.,1.8*maximum+6),g,.5)
    codes=np.unique(y[valid]); table=pg_table(grid,b,g,rn,int(y.max()),observed_codes=codes,device=device); logprior=np.log(prior_weights(grid,prior))
    llcodes=table['ll'][:,codes]
    likelihood={}; posterior={}; mle={}; variances={}; means={}
    for width in [5,9,15]:
        hist=histogram_at(y,valid,centers,width,codes); ll=hist@llcodes.T
        weights=ll+logprior; weights=np.exp(weights-logsumexp(weights,axis=1)[:,None])
        assert float(weights[:,-2:].sum(1).max())<1e-5,'Enlarge query posterior grid'
        likelihood[width]=ll; posterior[width]=weights; mle[width]=refined_mle(grid,ll)
        means[width]=weights@grid; variances[width]=np.maximum(weights@(grid**2)-means[width]**2,0)
    chosen=np.full(len(centers),15)
    conflict_9=np.abs(means[9]-means[15])>2*np.sqrt(variances[9]+variances[15])
    conflict_5=np.abs(means[5]-means[9])>2*np.sqrt(variances[5]+variances[9])
    chosen[conflict_9]=9; chosen[conflict_5]=5
    la=np.array([mle[w][i] for i,w in enumerate(chosen)])
    result={'LA-PG':la,**{f'L{w}-PG':mle[w] for w in [5,9,15]}}
    full_count=uniform_filter(valid.astype(float),5,mode='constant')*25
    lattice=np.zeros_like(valid); lattice[2::reference_stride,2::reference_stride]=True
    references=np.argwhere(lattice&(full_count>24.9999))
    if len(references):
        reference_hist=histogram_at(y,valid,references,5,codes); reference_ll=reference_hist@llcodes.T
        reference_weights=reference_ll+logprior
        reference_weights=np.exp(reference_weights-logsumexp(reference_weights,axis=1)[:,None])
        in_range=reference_weights[:,-2:].sum(1)<1e-5
        observed_matches=0


        for start in range(0,len(centers),256):
            query=centers[start:start+256]; similarity=np.sqrt(posterior[5][start:start+len(query)])@np.sqrt(reference_weights).T
            overlap=np.max(np.abs(query[:,None,:]-references[None,:,:]),axis=2)<5
            similarity[overlap]=-np.inf
            similarity[:,~in_range]=-np.inf
            order=np.argsort(-similarity,axis=1)
            for k in topks:
                if f'NL{k}-PG' not in result: result[f'NL{k}-PG']=np.empty(len(centers))
                for j in range(len(query)):
                    selected=order[j,:min(k,len(references))]
                    selected=selected[np.isfinite(similarity[j,selected])]
                    assert not overlap[j,selected].any()
                    observed_matches+=len(selected)
                    joint=likelihood[5][start+j]+reference_ll[selected].sum(0)
                    result[f'NL{k}-PG'][start+j]=refined_mle(grid,joint[None])[0]
    else:
        for k in topks: result[f'NL{k}-PG']=mle[5].copy()
    assert all(np.isfinite(v).all() and (v>=0).all() for v in result.values())
    return result,dict(candidates=len(centers),references=len(references),grid_points=len(grid),grid_max_e=float(grid[-1]),
        reference_stride=reference_stride,reference_window=5,query_reference_overlap_excluded=True,
        rejected_reference_grid_boundary=int((~in_range).sum()) if len(references) else 0,
        actual_selected_reference_count=observed_matches if len(references) else 0,
        distance='Bhattacharyya of normalized censored-PG likelihood times train-only prior',
        LA_window_counts={str(w):int((chosen==w).sum()) for w in [5,9,15]})
