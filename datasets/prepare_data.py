import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'datasets/source_manifest.json'
MATRIX=np.asarray([[.3485256854664918,.060970508902393306,.08471232558620155],[.08744703447690182,.700774429962975,.21177856954644453],[.02111332704585997,.10411145975775828,.4182996492763548]],np.float32)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def resolve_source(row,bsds_root,div2k_root):
    member=row['source_member'].replace('\\','/')
    root=bsds_root if row['family']=='BSDS500' else div2k_root
    return root.joinpath(*member.split('/'))

def preprocess(path,shape_wh):
    with Image.open(path) as image:
        if list(image.size)!=shape_wh: raise ValueError(f'Unexpected source shape: {path}')
        a=np.asarray(image.convert('RGB'),np.float32)/255
    if a.shape[0]>a.shape[1]: a=np.rot90(a).copy()
    a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
    a=np.maximum(a@MATRIX.T,0)
    h,w=a.shape[:2]
    scale=min(1,1024/max(h,w))
    size=(int(h*scale)//2*2,int(w*scale)//2*2)
    tensor=F.interpolate(torch.from_numpy(a).permute(2,0,1)[None],size=size,mode='bilinear',align_corners=False,antialias=True)
    return tensor[0].permute(1,2,0).contiguous().numpy()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bsds-root',type=Path,required=True)
    parser.add_argument('--div2k-root',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path(os.environ.get('SADPS_DATA',ROOT/'datasets/space_v2')))
    parser.add_argument('--cache-dir',type=Path,default=Path(os.environ.get('SADPS_CACHE',ROOT/'cache/space_v2_clean_v1')))
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    rows=[]
    for index,row in enumerate(manifest['rows'],1):
        src=resolve_source(row,args.bsds_root,args.div2k_root)
        if sha(src)!=row['sha256']: raise ValueError(f'Source SHA mismatch: {src}')
        cache=args.cache_dir/(row['cache_key']+'.npy')
        if not cache.exists():
            cache.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache.with_suffix('.npy.tmp')
            with tmp.open('wb') as f: np.save(f,preprocess(src,row['original_shape_wh']))
            os.replace(tmp,cache)
        row=dict(row,source_path=row['source_member'],cache_path=cache.name)
        rows.append(row)
        if index%50==0: print(f'{index}/{len(manifest["rows"])}',flush=True)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    payload=dict(policy=manifest['policy'],rows=rows,original_manifest_sha256=manifest['original_manifest_sha256'],split_seed=manifest['split_seed'],phash_hamming_threshold=manifest['phash_hamming_threshold'],counts=manifest['counts'],sealed_test_decoded=False)
    (args.data_dir/'MANIFEST.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','rows':len(rows),'counts':manifest['counts'],'data_manifest':str((args.data_dir/'MANIFEST.json').resolve()),'cache_dir':str(args.cache_dir.resolve())},indent=2))

if __name__=='__main__': main()
