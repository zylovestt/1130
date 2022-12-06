import torch
from NEW_AC import AC
from NEW_PUBLIC_ENV import RandomAgent,train_on_policy_agent,model_test,start_env
from NEW_rl_utils import mppp_train_on_policy_agent,mpp_train_on_policy_agent
import torch.multiprocessing as mp
import os
import numpy as np
import random

random.seed(81)
np.random.seed(76)
torch.manual_seed(31)

if __name__=='__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"

    env,anet,cnet,_,_,_,_,device,writer=start_env('cuda')
    aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
    coptim=torch.optim.NAdam(cnet.parameters(),lr=1e-3,eps=1e-8) # lr=1e-2
    # gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device
    # 0.96,0.95,1e-1,1e-1,1e-3
    agent=AC(0.95,0.95,1e-1,1e-1,1,anet,cnet,aoptim,coptim,device,writer)
    test_epochs=500
    # train_on_policy_agent(0,env,agent,100000,5,writer,200,test_epochs)
    mpp_train_on_policy_agent(0,env,agent,100000,5,1000,test_epochs) #mppp不是标准的同步策略,mpp貌似好些
    ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
    FTEST=lambda x:print(model_test(0,env,x,test_epochs))
    agent.explore=False
    FTEST(agent)
    FTEST(ra)
    # model_test(0,env,ra,1,True)