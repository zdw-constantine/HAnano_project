from torch import nn
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        if channel//reduction==0:
            reduction=1
        self.maxpool=nn.AdaptiveMaxPool1d(1)
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.se=nn.Sequential(
            nn.Conv1d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv1d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        
        max_result=self.maxpool(x) 
        avg_result=self.avgpool(x) 
        max_out=self.se(max_result) 
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out) 
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv1d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        # x:(B,C,H,W)
        max_result,_=torch.max(x,dim=1,keepdim=True)  
        avg_result=torch.mean(x,dim=1,keepdim=True)   
        result=torch.cat([max_result,avg_result],1)  
        output=self.conv(result)                    
        output=self.sigmoid(output)                  
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ChannelAttention=ChannelAttention(channel=channel,reduction=reduction)
        self.SpatialAttention=SpatialAttention(kernel_size=kernel_size)


    def forward(self, x):
        # (B,C,L)
        B,C,L = x.size()
        residual=x
        #print("Channel")
        out=x * self.ChannelAttention(x)    
        #print("Spatial")
        out=out * self.SpatialAttention(out)
        return out+residual



class CnnBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias,reduction,kernel_size_spa):
        super(CnnBlock,self).__init__()
        self.cnn=nn.Conv1d(in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride= stride, 
                padding=padding, 
                bias=bias)
        #print("bn")
        self.bn=nn.BatchNorm1d(out_channels)
        self.f=nn.SiLU()
        self.CBMA=CBAMBlock(channel=out_channels,reduction=reduction,kernel_size=kernel_size_spa)
        

    def forward(self,x):
        #print(x.shape)
        out = self.cnn(x)
        out = self.bn(out)
        #print(out.shape)
        out = self.f(out)
        #print(out.shape)
        out = self.CBMA(out)
        #print(out.shape)
        return out


class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Sequential(
            CnnBlock(
                in_channels = 1, 
                out_channels = 4, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True,
                reduction=1,
                kernel_size_spa=7),
            
            CnnBlock(
                in_channels = 4, 
                out_channels = 16, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True,
                reduction=2,
                kernel_size_spa=7),
            
            CnnBlock(
                in_channels = 16, 
                out_channels = 324, 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True,
                reduction=8,
                kernel_size_spa=7),
            
        )
        
        
    def forward(self,x):
        out=self.cnn(x)
        return out