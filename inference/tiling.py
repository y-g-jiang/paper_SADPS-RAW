import numpy as np
import torch

@torch.inference_mode()
def tile_predict(model,y,b,valid,device='cuda',core=256,halo=192,reverse=False):
    if core<16 or core%16 or halo<192 or halo%16: raise ValueError('core and halo must be multiples of 16; halo must be at least 192 packed pixels')
    h,w=y.shape[-2:]
    ph=(-h)%16;pw=(-w)%16
    yp=np.pad(y,((0,0),(0,ph),(0,pw)))
    vp=np.pad(valid,((0,0),(0,ph),(0,pw)))
    H,W=yp.shape[-2:]
    output=np.empty_like(yp,dtype=np.float32)
    tiles=[(r,c) for r in range(0,H,core) for c in range(0,W,core)]
    if reverse: tiles.reverse()
    bias=torch.as_tensor(b[None],device=device)
    for r,c in tiles:
        re=min(r+core,H);ce=min(c+core,W)
        top=max(0,r-halo);left=max(0,c-halo);bottom=min(H,re+halo);right=min(W,ce+halo)
        inp=torch.as_tensor(yp[None,:,top:bottom,left:right],device=device,dtype=torch.float32)
        mask=torch.as_tensor(vp[None,:,top:bottom,left:right],device=device,dtype=torch.float32)
        pred=model(inp,bias,mask)[0].float().cpu().numpy()
        output[:,r:re,c:ce]=pred[:,r-top:re-top,c-left:ce-left]
    return output[:,:h,:w]
