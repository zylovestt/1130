import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from NEW_ENV2 import Pro_Flow,Job_Flow,NEW_ENV,RandomAgent
from NEW_NET import CriticNet,QNet,QNet2
from copy import deepcopy
from NEW_rl_utils import train_on_policy_agent,train_off_policy_agent,ReplayBuffer,NEW_ReplayBuffer,Quick_ReplayBuffer
from NEW_TEST import model_test
import random
import torch

random.seed(81)
np.random.seed(71)
torch.manual_seed(51)
device='cuda'
writer=SummaryWriter(comment='NEW_ENV')
pro_config={'c':(10,2),'r':(50,10),'v':(10,2)}
# pro_config={'c':(2,20),'r':(10,100),'v':(5,20)}
pro_num=8
pro_config['num_pro']=0.9
# pro_config['num_pro']=np.ones(pro_num)/pro_num
# pro_config['num_pro']=[0,0,0,0,1]
PF=Pro_Flow(1,0,pro_config,pro_num,writer,True)
jc={'k':0.5,'r':(100,10),'loc_mean':(-100,100),'loc_scale':1,'time':(20,0)}
# jc={'k':0.5,'r':(10,200),'loc_mean':(-100,100),'loc_scale':1,'time':(20,0)}
tasknum=8
env_steps=100
JF=Job_Flow(0,jc,tasknum,env_steps)
JF.cal_mean_scale(1000)
env=NEW_ENV(PF,JF,env_steps,writer)
# print(env.pf.pros.ps)
env.normalize(100)
state=env.reset()

# anet=QNet(state.size,1,500,2,500,env.pros.num,env.jf.tasknum).to(device)
# cnet=CriticNet(state.size,3,500).to(device)
# qnet=QNet(state.size,1,300,1,100,env.pros.num,env.jf.tasknum).to(device)

# td3_anet=QNet(state.size,1,500,2,500,env.pros.num,env.jf.tasknum).to(device)
# td3_qnet1=CriticNet(state.size+env.pros.num*env.jf.tasknum,3,500).to(device)
# td3_qnet2=CriticNet(state.size+env.pros.num*env.jf.tasknum,3,500).to(device)



anet=QNet2(state.size,3,500,env.pros.num,env.jf.tasknum).to(device)
cnet=CriticNet(state.size,3,500).to(device)
qnet=QNet2(state.size,3,500,env.pros.num,env.jf.tasknum).to(device)

td3_anet=QNet2(state.size,3,500,env.pros.num,env.jf.tasknum).to(device)
td3_qnet1=CriticNet(state.size+env.pros.num*env.jf.tasknum,3,500).to(device)
td3_qnet2=CriticNet(state.size+env.pros.num*env.jf.tasknum,3,500).to(device)