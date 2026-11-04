import torch
from torch import nn
from torch.nn import functional as F
from .common import G,RN

class Norm(nn.Module):
    def __init__(self,c):
        super().__init__(); self.weight=nn.Parameter(torch.ones(1,c,1,1)); self.bias=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):

        dtype=x.dtype; z=x.float(); z=(z-z.mean(1,keepdim=True))*torch.rsqrt(z.var(1,keepdim=True,unbiased=False)+1e-6)
        return (z*self.weight+self.bias).to(dtype)

class Block(nn.Module):
    def __init__(self,c,sca):
        super().__init__(); self.n1=Norm(c); self.n2=Norm(c)
        self.a=nn.Conv2d(c,c*2,1); self.dw=nn.Conv2d(c*2,c*2,3,padding=1,groups=c*2)
        self.sca=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c,1)) if sca else None
        self.b=nn.Conv2d(c,c,1); self.c=nn.Conv2d(c,c*2,1); self.d=nn.Conv2d(c,c,1)
        self.beta=nn.Parameter(torch.zeros(1,c,1,1)); self.gamma=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        a,b=self.dw(self.a(self.n1(x))).chunk(2,1); z=a*b
        if self.sca is not None: z=z*self.sca(z)
        x=x+self.beta*self.b(z); a,b=self.c(self.n2(x)).chunk(2,1)
        return x+self.gamma*self.d(a*b)

class Spatial(nn.Module):
    def __init__(self,c):
        super().__init__(); self.norm=Norm(c); self.qkv=nn.Conv2d(c,c*3,1); self.proj=nn.Conv2d(c,c,1)
        self.coords=nn.Conv2d(2,c,1,bias=False); self.gate=nn.Parameter(torch.zeros(1,c,1,1)); self.c=c
    def forward(self,x):
        b,c,h,w=x.shape
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),torch.linspace(-1,1,w,device=x.device,dtype=x.dtype),indexing='ij')
        z=self.norm(x)+self.coords(torch.stack((xx,yy))[None])
        q,k,v=self.qkv(z).reshape(b,3,4,c//4,h*w).permute(1,0,2,4,3).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v).transpose(-1,-2).reshape(b,c,h,w)
        return x+torch.tanh(self.gate)*self.proj(a)

class ComputeMatched(nn.Module):
    def __init__(self,c):
        super().__init__(); self.norm=Norm(c)
        self.path=nn.Sequential(nn.Conv2d(c,2*c,1),nn.GELU(),nn.Conv2d(2*c,c,1))
        self.gate=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x): return x+torch.tanh(self.gate)*self.path(self.norm(x))

class Network(nn.Module):
    def __init__(self,kind='N0',head='PG'):
        super().__init__(); assert kind in ['N0','N1','N2','N2C']; self.kind=kind; self.head_type=head
        sca=kind!='N0'; self.stem=nn.Conv2d(16,32,3,padding=1)
        self.enc=nn.ModuleList(); self.down=nn.ModuleList(); self.up=nn.ModuleList(); self.dec=nn.ModuleList()
        c=32
        for count in [2,2,4,4]:
            self.enc.append(nn.Sequential(*[Block(c,sca) for _ in range(count)])); self.down.append(nn.Conv2d(c,2*c,2,stride=2)); c*=2
        self.middle=nn.Sequential(*[Block(c,sca) for _ in range(4)])
        self.extra=Spatial(c) if kind=='N2' else ComputeMatched(c) if kind=='N2C' else nn.Identity()
        for _ in range(4):
            self.up.append(nn.Sequential(nn.Conv2d(c,2*c,1,bias=False),nn.PixelShuffle(2))); c//=2
            self.dec.append(nn.Sequential(Block(c,sca),Block(c,sca)))
        self.head=nn.Conv2d(c,4 if head=='PG' else 4*66,1)
    def forward(self,y,b,valid):
        assert y.ndim==4 and y.shape[1]==4 and y.shape[-1]%16==0 and y.shape[-2]%16==0
        bm=b.expand_as(y); v=valid.expand(y.shape[0],1,*y.shape[-2:])
        constants=torch.cat([torch.full_like(v,G),torch.full_like(v,RN),torch.zeros_like(v)],1)
        x=torch.cat((torch.asinh(y/4),((y==0)&(v>0)).to(y.dtype),bm,v,constants),1)
        x=self.stem(x); skips=[]
        for enc,down in zip(self.enc,self.down): x=enc(x); skips.append(x); x=down(x)
        x=self.extra(self.middle(x))
        for up,dec,skip in zip(self.up,self.dec,skips[::-1]): x=dec(up(x)+skip)
        z=self.head(x).float()
        if self.head_type=='PG': return F.softplus(z)
        return z.reshape(z.shape[0],4,66,*z.shape[-2:]).permute(0,1,3,4,2).log_softmax(-1)
