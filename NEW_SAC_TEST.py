import torch
from NEW_SAC import SAC
from NEW_PUBLIC_ENV import RandomAgent,ReplayBuffer,train_off_policy_agent,model_test,env,anet,qnet,qnet1,device,writer

aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-5,eps=1e-8) # lr=1e-4
qoptim=torch.optim.NAdam(qnet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
qoptim1=torch.optim.NAdam(qnet1.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4 
# anet,qnet1,qnet2,aoptim,qoptim1,qoptim2,alpha_lr, target_entropy, tau, gamma,  device, writer
# 0.96,0.95,1e-1,1e-1,1e-3
agent=SAC(anet,qnet,qnet1,aoptim,qoptim,qoptim1,1e-2,0.1,0.5,0.96,device,writer)
replay_buffer = ReplayBuffer(10000)

return_list = train_off_policy_agent(9,env, agent, 500, replay_buffer, 1000, 32)
ra=RandomAgent(9,env.pros.num,env.jf.tasknum) 
FTEST=lambda x:print(model_test(0,env,x,20))
FTEST(agent)
FTEST(ra)  