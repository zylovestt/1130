import torch
import time
from NEW_TD3 import TD3
from NEW_PUBLIC_ENV import RandomAgent,Quick_ReplayBuffer,ReplayBuffer,NEW_ReplayBuffer,train_off_policy_agent,model_test,start_env
from NEW_rl_utils import mp_train_off_policy_agent,mpp_train_off_policy_agent,mppp_train_off_policy_agent
import torch.multiprocessing as mp
import os
import random
import numpy as np

random.seed(851)
np.random.seed(761)
torch.manual_seed(531)

if __name__=='__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    start=time.time()
    env,_,_,_,td3_anet,td3_qnet1,td3_qnet2,device,writer=start_env()
    print(device)
    aoptim=torch.optim.NAdam(td3_anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
    qoptim1=torch.optim.NAdam(td3_qnet1.parameters(),lr=1e-3,eps=1e-8) # lr=1e-3
    qoptim2=torch.optim.NAdam(td3_qnet2.parameters(),lr=1e-3,eps=1e-8) # lr=1e-3
    # anet,qnet1,qnet2,aoptim,qoptim1,qoptim2, tau, gamma, device,writer
    agent=TD3(td3_anet,td3_qnet1,td3_qnet2,aoptim,qoptim1,qoptim2,1e-2,0.95,device,writer,1e-1)
    replay_buffer = Quick_ReplayBuffer(100000,device,env.state_size,env.action_size)
    test_cycles=1000
    test_epochs=50
    return_list=mppp_train_off_policy_agent(0,env,agent,100000,replay_buffer,10000,1024,10,test_cycles,test_epochs)
    print('start test')
    ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
    FTEST=lambda x:model_test(0,env,x,test_epochs)
    agent.explore=False
    print('agent:',FTEST(agent))
    print('random:',FTEST(ra))
    print('time:',time.time()-start)