import torch
import time
from NEW_MULTI_SAC import MULTI_SAC
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
    env,anet,_,qnet,td3_anet,_,_,device,writer,conn,curs,date_time=start_env('cuda')
    print(device)
    aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
    qoptim1=torch.optim.NAdam(qnet.parameters(),lr=1e-3,eps=1e-8) # lr=1e-3
    qoptim2=torch.optim.NAdam(td3_anet.parameters(),lr=1e-3,eps=1e-8) # lr=1e-3
    # anet,qnet1,qnet2,aoptim,qoptim1,qoptim2, tau, gamma, device,writer
    agent=MULTI_SAC(anet,qnet,td3_anet,aoptim,qoptim1,qoptim2,5e-3,0.95,device,0.5,1e-2,1e-1,conn,curs,date_time)
    replay_buffer = Quick_ReplayBuffer(100000,device,env.state_size,env.action_size)
    test_model=False
    if test_model: 
        agent.load_model('sac_actor')
    else:
        test_cycles=1000
        test_epochs=500
        return_list=mppp_train_off_policy_agent(0,env,agent,50000,replay_buffer,10000,1024,10,test_cycles,test_epochs)
        agent.save_model('sac_actor')
        ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
        FTEST=lambda x:model_test(0,env,x,test_epochs)
        # agent.explore=False
        print('start test')
        print('agent:',FTEST(agent))
        print('random:',FTEST(ra))
    model_test(137,env,agent,1,True)
    print('time:',time.time()-start)
    conn.commit()
    curs.close()
    conn.close()