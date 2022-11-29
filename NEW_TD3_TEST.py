import torch
import time
from NEW_TD3 import TD3
from NEW_PUBLIC_ENV import RandomAgent,Quick_ReplayBuffer,ReplayBuffer,NEW_ReplayBuffer,train_off_policy_agent,model_test,env,td3_anet,td3_qnet1,td3_qnet2,device,writer

start=time.time()
aoptim=torch.optim.NAdam(td3_anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
qoptim1=torch.optim.NAdam(td3_qnet1.parameters(),lr=1e-3,eps=1e-8) # lr=1e-4
qoptim2=torch.optim.NAdam(td3_qnet2.parameters(),lr=1e-3,eps=1e-8) # lr=1e-4
# anet,qnet1,qnet2,aoptim,qoptim1,qoptim2, tau, gamma, device,writer
# 0.96,0.95,1e-1,1e-1,1e-3
agent=TD3(td3_anet,td3_qnet1,td3_qnet2,aoptim,qoptim1,qoptim2,1e-2,0.95,device,writer,1e-1)
replay_buffer = Quick_ReplayBuffer(100000,device)
test_cycles=100
test_epochs=50
return_list = train_off_policy_agent(0,env, agent, 1000, replay_buffer, 10000, 1024,10,test_cycles,test_epochs)
ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
FTEST=lambda x:print(model_test(0,env,x,test_epochs))
agent.explore=False
FTEST(agent) 
FTEST(ra)
print('time:',time.time()-start)