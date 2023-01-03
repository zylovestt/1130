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

    env,anet,cnet,_,_,_,_,device,writer,conn,curs,date_time=start_env('cuda')
    aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
    coptim=torch.optim.NAdam(cnet.parameters(),lr=1e-3,eps=1e-8) # lr=1e-2
    # gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device
    # 0.96,0.95,1e-1,1e-1,1e-3
    agent=AC(0.95,0.95,1e-1,1e-1,1e-3,anet,cnet,aoptim,coptim,device,writer,conn,curs,date_time) 
    test_epochs=500

    # # train_on_policy_agent(0,env,agent,100000,5,writer,200,test_epochs)
    # mpp_train_on_policy_agent(0,env,agent,50000,10,1000*10,test_epochs) #mppp不是标准的同步策略,mpp貌似好些

    update_steps=10
    # train_on_policy_agent(0,env, agent, 10000,5,writer,200,test_epochs)
    test_model=False
    if test_model: 
        agent.load_model('ac_actor')
    else:
        test_cycles=1000
        test_epochs=500
        return_list=mppp_train_on_policy_agent(0,env, agent, 50000,update_steps,test_cycles,test_epochs)
        agent.save_model('ac_actor')
        ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
        FTEST=lambda x:model_test(0,env,x,test_epochs)
        # agent.explore=False
        print('start test')
        print('agent:',FTEST(agent))
        print('random:',FTEST(ra))
    model_test(137,env,agent,1,True)