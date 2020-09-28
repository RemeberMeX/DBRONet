import torch
import torch.nn as nn

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

#Multi-scale residual block
class MRN(nn.Module):
    def __init__(self,nin=64,use_GPU=True):
        super(MRN, self).__init__()
        self.use_GPU = use_GPU
        ksize1 = 3
        ksize2 = 5
        pad1=int((ksize1-1)/2)
        pad2=int((ksize2-1)/2)
        self.conv1 = nn.Conv2d(nin, nin, ksize1, 1,pad1)
        self.conv2 = nn.Conv2d(nin, nin, ksize2, 1,pad2)
        self.conv3 = nn.Conv2d(nin, nin, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = input
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x))

        y1 = self.conv1(x1 + x2 )
        y2 = self.conv2(x1 + x2 )

        y1 = self.lrelu(y1)
        y2 = self.lrelu(y2)

        out = self.conv3(y1 + y2 )
        out =  input+out
        return out

class Network(nn.Module):
    def __init__(self,nin=64,use_GPU=True):
        super(Network,self).__init__()
        self.use_GPU = use_GPU
        self.conv1=nn.Sequential(nn.Conv2d(3,64,1,1,0))

        self.block1=nn.Sequential(nn.Conv2d(nin,nin,3,1,padding=1),
                                 nn.LeakyReLU(0.2))
        #Multi-scale residual block
        self.block2=nn.Sequential(MRN(nin),
                                nn.LeakyReLU(0.2))

        self.block3=nn.Sequential(nn.Conv2d(nin,3,3,1,padding=1))
        #dilated convolution and d=2
        self.block4=nn.Sequential(nn.Conv2d(nin,nin,3,1,padding=2,dilation=2),
                                nn.LeakyReLU(0.2))
        self.se=SEblock(64,16)
    def forward(self,input):
        #the upper branch multi-scale rain streak extraction block(MRSEB)
        x1=self.conv1(input)
        x1=self.block1(x1)
        for j in range(2):
            x11=x1
            for i in range(3):
                x1=self.block2(x1)
                x1=self.block1(x1)
            x1=x11+x1
            x1=self.block1(x1)
        x1=self.se(x1)
        x1=self.block3(x1)#64->3
        out1=input+x1
        #the lower branch dilated convolution attention residual block (DARB)
        x2=self.conv1(input)
        x2=self.block1(x2)
        for j in range(2):
            x21=x2
            for i in range(5):
                x2=self.block4(x2)
            x2=self.se(x2)
            x2=x2+x21
            x2=self.block1(x2)
        x2=self.block3(x2)#64->3
        out2=input+x2
        #residual overlay
        out=out1+out2
        out=self.conv1(out)
        out=self.se(out)
        out=self.block3(out)
        out=input+out
        return out,out1,out2

#SE Block
class SEblock(nn.Module):
    def __init__(self,nin=64,reduces=16):
        super(SEblock,self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(nn.Linear(nin,nin // reduces,bias=False),
                            nn.ReLU(inplace=True),
                            nn.Linear(nin//reduces,nin,bias=False),
                            nn.Sigmoid())   
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)