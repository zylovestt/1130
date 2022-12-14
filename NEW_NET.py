import torch
import torch.nn as nn
import math

# FMakeNet=lambda x,y,z:[nn.PReLU() if not i%2 else nn.Linear(x,x) if i<y*2-1 else nn.Linear(x,z) for i in range(y*2)]

def FMakeNet(input,middle,output,deep):
    layer=[]
    # assert not (deep==1 and not(middle==output)),'wrong'
    if deep==1:
        middle=output
    for i in range(deep):
        layer.append(nn.PReLU())
        # layer.append(nn.ReLU())
        # layer.append(nn.SELU())
        if i==0:
            layer.append(nn.Linear(input,middle))
        elif i==deep-1:
            layer.append(nn.Linear(middle,output))
        else:
            layer.append(nn.Linear(middle,middle))
    return layer

class CriticNet(nn.Module):
    def __init__(self,input_size,base_deep,base_width):
        super().__init__()
        self.base_net=nn.Sequential(nn.Linear(input_size,base_width),*FMakeNet(base_width,base_width,1,base_deep-1))
    
    def __call__(self,x:torch.tensor):
        return self.base_net(x)
    
    # def _init_weights(self, m):
    #     nn.init.kaiming_normal_(m.weight)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     # 也可以判断是否为conv2d，使用相应的初始化方式 
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     # 是否为批归一化层
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

class QNet(nn.Module):
    def __init__(self,input_size,base_deep,base_width,top_deep,top_width,output_size,outnet_num):
        super().__init__()
        assert top_deep
        self.pros=output_size
        self.base_net=FMakeNet(base_width,base_width,base_width,base_deep-1)
        if len(self.base_net):
            self.base_net=nn.Sequential(nn.Linear(input_size,base_width),*self.base_net)
        else:
            self.base_net=None
            base_width=input_size
        self.top_net=nn.ModuleList([nn.Sequential(*FMakeNet(base_width,top_width,output_size,top_deep)) for _ in range(outnet_num)])
        self.output_size=output_size
        self.outnet_num=outnet_num
    
    def __call__(self,x:torch.tensor):
        if self.base_net is None:
            bx=x
        else:
            bx=self.base_net(x)
        out=torch.cat([f(bx).unsqueeze(0) for f in self.top_net],dim=0).transpose(0,1)
        mask=x[:,:self.pros].unsqueeze(1)
        a=torch.zeros_like(mask)
        a[:]=1e8
        a*=(~(mask==1))
        return out-a
    
    # def _init_weights(self, m):
    #     nn.init.kaiming_normal_(m.weight)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     # 也可以判断是否为conv2d，使用相应的初始化方式 
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     # 是否为批归一化层
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

class QNet2(nn.Module):
    def __init__(self,input_size,deep,width,pro_num,task_num):
        super().__init__()
        self.task_num=task_num
        self.pro_num=pro_num
        self.net=nn.Sequential(nn.Linear(input_size,width),*FMakeNet(width,width,task_num*pro_num,deep-1))
    
    def __call__(self,x:torch.tensor):
        out=self.net(x).reshape(x.shape[0],self.task_num,-1)
        mask=x[:,:self.pro_num].unsqueeze(1)
        a=torch.zeros_like(mask)
        a[:]=1e8
        a*=(~(mask>0))
        return out-a
    
    # def _init_weights(self, m):
    #     nn.init.kaiming_normal_(m.weight)

    # def weight_init(self,m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     # 也可以判断是否为conv2d，使用相应的初始化方式 
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     # 是否为批归一化层
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)


    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()