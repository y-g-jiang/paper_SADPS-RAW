import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from minimal_source.network import Network
from minimal_source.common import G, RN, K
from minimal_source.law import exact
from minimal_source.sampling import sample
from tiling import tile_predict

def run(input_path,output_path,device='cuda',seed=7103,draw=0,image_id=12800,top=0,left=0,frame_width=None,core=256,halo=192,reverse_tiles=False):
    device=torch.device(device)
    data=np.load(input_path)
    y=np.asarray(data['Y'])
    b=np.asarray(data['b'],dtype=np.float32).reshape(4,1,1)
    if y.ndim!=3 or y.shape[0]!=4 or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>15871):
        raise ValueError('Y must be a four-plane nonnegative integer Bayer array [R,G1,G2,B]')
    h,w=y.shape[1:]
    if not np.isfinite(b).all() or np.any(np.abs(b)>.25001): raise ValueError('b must be finite and within the trained range [-0.25,0.25] DN')
    if h%16 or w%16: raise ValueError('Packed input dimensions must be multiples of 16')
    if top<0 or left<0 or top%2 or left%2: raise ValueError('Mosaic top/left must be nonnegative even coordinates')
    frame_width=int(left+2*w if frame_width is None else frame_width)
    if frame_width<left+2*w: raise ValueError('frame width is smaller than the input extent')
    valid=np.asarray(data['valid'],dtype=np.float32) if 'valid' in data else np.ones((1,h,w),np.float32)
    if valid.shape==(h,w): valid=valid[None]
    if valid.shape!=(1,h,w): raise ValueError('valid must have shape [H,W] or [1,H,W]')
    model_path=ROOT/'BEST_WEIGHTS.pt'
    expected="662eef3e581eb77d1e948ee800d6e63c0ab0039c3b4cff8f9fafe8a99074894f"
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest!=expected: raise ValueError('Frozen model SHA mismatch')
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=False)
    if checkpoint.get('step')!=80000 or checkpoint.get('config',{}).get('seed')!=7103: raise ValueError('Unexpected checkpoint identity')
    torch.set_num_threads(2)
    model=Network('N0','PG').to(device).eval()
    model.load_state_dict(checkpoint['model'])
    signal=tile_predict(model,y,b,valid,device=device,core=core,halo=halo,reverse=reverse_tiles)
    mask=(y==0)&(valid[0][None]>0)
    c,yy,xx=np.nonzero(mask)
    ids=((top+2*yy+c//2)*frame_width+(left+2*xx+c%2)).astype(np.uint64)
    chosen=np.empty(len(c),np.int16)
    zero_prob=np.zeros((4,h,w),np.float32)
    for start in range(0,len(c),512):
        sl=slice(start,start+512);cc=c[sl];rr=yy[sl];col=xx[sl]
        s=signal[cc,rr,col].astype(np.float64);bias=b[cc,0,0].astype(np.float64)
        with torch.inference_mode():
            logq=exact(torch.as_tensor(s,device=device),torch.as_tensor(bias,device=device)).cpu().numpy()
        logq-=logsumexp(logq,axis=1)[:,None]
        zero_prob[cc,rr,col]=np.exp(logq[:,0]).astype(np.float32)
        chosen[sl]=sample(logq,s,bias,ids[sl],seed=seed,draw=draw,image_id=image_id)
    signed=y.astype(np.int16).copy()
    signed[c,yy,xx]=chosen
    if not np.array_equal(signed[y>0],y[y>0]) or not np.array_equal(np.maximum(signed,0),y):
        raise AssertionError('positive-code identity or re-clipping invariant failed')
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,signal_DN=signal,zero_probability=zero_prob,signed=signed)
    return {'status':'PASS','checkpoint_sha256':digest,'step':checkpoint['step'],'seed':7103,'device':str(device),'gain_DN_per_e':G,'analog_RN_DN':RN,'K':K,'zero_candidates':int(len(c)),'sampled_negative':int(np.sum(chosen<0)),'positive_codes_preserved':True,'reclip_exact':True,'output':str(output_path)}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=7103);p.add_argument('--draw',type=int,default=0);p.add_argument('--image-id',type=int,default=12800);p.add_argument('--top',type=int,default=0);p.add_argument('--left',type=int,default=0);p.add_argument('--frame-width',type=int);p.add_argument('--core',type=int,default=256);p.add_argument('--halo',type=int,default=192);p.add_argument('--reverse-tiles',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.input,a.output,a.device,a.seed,a.draw,a.image_id,a.top,a.left,a.frame_width,a.core,a.halo,a.reverse_tiles),indent=2))
