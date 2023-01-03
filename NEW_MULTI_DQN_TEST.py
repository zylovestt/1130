import torch
import time
from NEW_MULTI_DQN import MULTI_DQN
from NEW_PUBLIC_ENV import RandomAgent,Quick_ReplayBuffer,ReplayBuffer,NEW_ReplayBuffer,train_off_policy_agent,model_test,start_env
from NEW_rl_utils import mp_train_off_policy_agent,mpp_train_off_policy_agent,mppp_train_off_policy_agent
import torch.multiprocessing as mp
import os
import random
import numpy as np

random.seed(81)
np.random.seed(76)
torch.manual_seed(31)

if __name__=='__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    start=time.time()
    env,_,_,_,qnet,_,_,device,writer,conn,curs,date_time=start_env('cuda')
    print(device)
    qoptim=torch.optim.NAdam(qnet.parameters(),lr=1e-3,eps=1e-8) # lr=1e-4
    agent=MULTI_DQN(0.95,qnet,qoptim,5e-3,device,1e-1,conn,curs,date_time)
    replay_buffer = Quick_ReplayBuffer(100000,device,env.state_size,env.action_size) 
    test_model=False
    if test_model: 
        agent.load_model('dqn_actor')
    else:
        test_cycles=1000
        test_epochs=500
        return_list=mppp_train_off_policy_agent(0,env,agent,50000,replay_buffer,10000,1024,10,test_cycles,test_epochs)
        agent.save_model('dqn_actor')
        ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
        FTEST=lambda x:model_test(0,env,x,test_epochs)
        # agent.explore=False
        print('start test')
        print('agent:',FTEST(agent))
        print('random:',FTEST(ra))
    model_test(137,env,agent,1,True)