import time
import numpy as np
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from NEW_ENV2 import Pro_Flow,Job_Flow,NEW_ENV,RandomAgent
from NEW_NET import CriticNet,QNet,QNet2
from copy import deepcopy
from NEW_rl_utils import train_on_policy_agent,train_off_policy_agent,ReplayBuffer,NEW_ReplayBuffer,Quick_ReplayBuffer
from NEW_TEST import model_test
import random
import torch.nn as nn
from pprint import pprint
import sqlite3

# print(__name__)

# def weight_init(m):
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

def start_env(device):
    device=device
    # writer=SummaryWriter(comment='NEW_ENV')
    writer=None
    # pro_config={'c':(10,2),'r':(50,10),'v':(10,2)}
    pro_config={'c':(1,0.15),'r':(5,0.75),'v':(0.1,0.015),'lx':(-1,1),'ly':(-1,1),'alpha':(0,300),'beta':(0,50)}
    # pro_config={'c':(2,20),'r':(10,100),'v':(5,20)}
    pro_num=8
    pro_config['num_pro']=0.5
    # pro_config['num_pro']=np.ones(pro_num)/pro_num
    # pro_config['num_pro']=[0,0,0,0,1]
    PF=Pro_Flow(1,0,pro_config,pro_num,writer,True)
    # jc={'k':0.5,'r':(10,2),'loc_mean':(-10,10),'loc_scale':0.1,'time':(2,0)} #2,0???
    jc={'k':0.5,'r':(1,0.15),'loc_mean':(-1,1),'loc_scale':1,'time':(5,0)} #2,0???
    # jc={'k':0.5,'r':(10,200),'loc_mean':(-100,100),'loc_scale':1,'time':(20,0)}
    tasknum=8
    env_steps=100
    JF=Job_Flow(0,jc,tasknum,env_steps)
    # JF.cal_mean_scale(1000)
    env=NEW_ENV(PF,JF,env_steps,writer)
    pprint(env.pf.pros.ps)

    # env.normalize(1000)
    # np.save('md_li_ab1',env.md)
    # np.save('sd_li_ab1',env.sd)
    # env.md=np.load('md_li_ab0.5.npy')
    # env.sd=np.load('sd_li_ab0.5.npy')
    env.md=np.load('md_li_ab1.npy')
    env.sd=np.load('sd_li_ab1.npy')
    print('md',env.md,'sd',env.sd)

    # anet=QNet(env.state_size,1,500,2,500,env.pros.num,env.jf.tasknum).to(device)
    # cnet=CriticNet(env.state_size,3,500).to(device)
    # qnet=QNet(env.state_size,1,300,1,100,env.pros.num,env.jf.tasknum).to(device)

    # td3_anet=QNet(env.state_size,1,500,2,500,env.pros.num,env.jf.tasknum).to(device)
    # td3_qnet1=CriticNet(env.state_size+env.pros.num*env.jf.tasknum,3,500).to(device)
    # td3_qnet2=CriticNet(env.state_size+env.pros.num*env.jf.tasknum,3,500).to(device)



    anet=QNet2(env.state_size,3,500,env.pros.num,env.jf.tasknum).to(device)
    cnet=CriticNet(env.state_size,3,500).to(device)
    qnet=QNet2(env.state_size,3,500,env.pros.num,env.jf.tasknum).to(device)

    td3_anet=QNet2(env.state_size,3,500,env.pros.num,env.jf.tasknum).to(device)
    
    td3_qnet1=CriticNet(env.state_size+env.action_size,3,500).to(device)
    td3_qnet2=CriticNet(env.state_size+env.action_size,3,500).to(device)

    # for net in [anet,cnet,qnet,td3_anet,td3_qnet1,td3_qnet2]:
    #     net.apply(weight_init)

    print(anet)
    print(cnet)
    print(qnet)
    print(td3_anet)
    print(td3_qnet1)
    print(td3_qnet2)

    conn=sqlite3.connect('record.db')
    curs=conn.cursor()
    date_time=time.strftime('%Y-%m-%d %H:%M:%S')
    return env,anet,cnet,qnet,td3_anet,td3_qnet1,td3_qnet2,device,writer,conn,curs,date_time

if __name__=='__main__':
    import time
    print(time.strftime('%Y.%m.%d',time.localtime(time.time())))
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    conn=sqlite3.connect('record.db')
    curs=conn.cursor()
    t=time.strftime('%Y-%m-%d %H:%M:%S')
    print(type(t))
    print("insert into recordvalue values(%s,'td3','acloss',0,0.334)"%t)
    curs.execute("insert into recordvalue values('%s','td3','acloss',0,0.334)"%t)
    conn.commit()
    # curs.execute('select getdate()')
    # rows=curs.fetchone()
    # print(rows)
    curs.execute('drop table recordvalue')
    try:
        curs.execute('''create table recordvalue
                        (date datetime,
                        algorithm varchar(20),
                        recordname varchar(20),
                        step int,
                        recordsize float,
                        primary key(date,algorithm,recordname,step))''')
    except Exception as e:
        print(repr(e))