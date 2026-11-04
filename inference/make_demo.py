import argparse
from pathlib import Path
import numpy as np
G=21.601092412099675
RN=24.95960511926749
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='demo_input.npz');p.add_argument('--seed',type=int,default=7103);p.add_argument('--size',type=int,default=128);a=p.parse_args()
    if a.size%16: raise ValueError('size must be a multiple of 16')
    rng=np.random.default_rng(a.seed);yy,xx=np.mgrid[:a.size,:a.size];planes=[]
    for c in range(4):
        texture=5+18*(xx/a.size)+8*(yy/a.size)+4*np.sin((xx+2*yy+c*7)/9)
        texture=np.maximum(texture,0);signal=G*rng.poisson(texture/G)+rng.normal(0,RN,texture.shape)
        planes.append(np.maximum(np.floor(signal+.5),0).astype(np.uint16))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,Y=np.stack(planes),b=np.zeros(4,np.float32),valid=np.ones((1,a.size,a.size),np.float32))
    print(a.output)
if __name__=='__main__':main()
