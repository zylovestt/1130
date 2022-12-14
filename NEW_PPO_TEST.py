import torch
from NEW_PPO import PPO
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
    # eps,epochs,gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device
    # 0.8,3,0.96,0.95,1e-1,1e-1,1e-3
    agent=PPO(0.2,5,0.95,0.95,1e-1,1e-1,1e-4,anet,cnet,aoptim,coptim,device,writer,conn,curs,date_time)

    # train_on_policy_agent(9,env, agent, 500,10,writer)
    # ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
    # FTEST=lambda x:print(model_test(0,env,x,20))
    # agent.explore=False
    # FTEST(agent)
    # FTEST(ra)
    # update_steps=10
    # # train_on_policy_agent(0,env, agent, 10000,5,writer,200,test_epochs)
    # mpp_train_on_policy_agent(0,env, agent, 50000*update_steps,update_steps,1000*update_steps,test_epochs)

    update_steps=10
    test_model=False
    if test_model: 
        agent.load_model('ppo_actor')
    else:
        test_cycles=1000
        test_epochs=500
        return_list=mppp_train_on_policy_agent(0,env, agent, 50000,update_steps,test_cycles,test_epochs)
        agent.save_model('ppo_actor')
        ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
        FTEST=lambda x:model_test(0,env,x,test_epochs)
        # agent.explore=False
        print('start test')
        print('agent:',FTEST(agent))
        print('random:',FTEST(ra))
    model_test(137,env,agent,1,True)