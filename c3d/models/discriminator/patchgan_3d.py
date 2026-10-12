from __future__ import annotations
from torch import nn

class PatchDiscriminator3D(nn.Module):
    def __init__(self,in_channels=1,base_channels=32,layers=4,max_channels=256):
        super().__init__(); mods=[nn.Conv3d(in_channels,base_channels,4,2,1),nn.LeakyReLU(.2,inplace=True)];c=base_channels
        for i in range(1,int(layers)):
            nc=min(int(max_channels),c*2);stride=1 if i==int(layers)-1 else 2
            mods += [nn.Conv3d(c,nc,4,stride,1,bias=False),nn.GroupNorm(1,nc),nn.LeakyReLU(.2,inplace=True)];c=nc
        mods.append(nn.Conv3d(c,1,3,1,1));self.net=nn.Sequential(*mods)
    def forward(self,x):return self.net(x)
