import torch
from NEW_PPO_PENLTY import PPO
from NEW_PUBLIC_ENV import RandomAgent,train_on_policy_agent,model_test,env,anet,cnet,device,writer
aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
coptim=torch.optim.NAdam(cnet.parameters(),lr=1e-2,eps=1e-8) # lr=1e-2
# eps,epochs,gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device
# 0.8,3,0.96,0.95,1e-1,1e-1,1e-3
agent=PPO(3,5,0.95,0.7,1e-1,1e-1,1e-4,anet,cnet,aoptim,coptim,device,writer)
train_on_policy_agent(9,env, agent, 200,10,writer)
ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
FTEST=lambda x:print(model_test(0,env,x,20))
FTEST(agent)
FTEST(ra)