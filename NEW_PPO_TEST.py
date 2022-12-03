import torch
from NEW_PPO import PPO
from NEW_PUBLIC_ENV import RandomAgent,train_on_policy_agent,model_test,start_env

env,anet,cnet,_,_,_,_,device,writer=start_env()
aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
coptim=torch.optim.NAdam(cnet.parameters(),lr=1e-3,eps=1e-8) # lr=1e-2
# eps,epochs,gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device
# 0.8,3,0.96,0.95,1e-1,1e-1,1e-3
agent=PPO(0.2,5,0.95,0.95,1e-1,1e-1,0.5*1e-3,anet,cnet,aoptim,coptim,device,writer)

# train_on_policy_agent(9,env, agent, 500,10,writer)
# ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
# FTEST=lambda x:print(model_test(0,env,x,20))
# agent.explore=False
# FTEST(agent)
# FTEST(ra)

test_epochs=50
train_on_policy_agent(0,env, agent, 1000,5,writer,100,test_epochs)
ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
FTEST=lambda x:print(model_test(0,env,x,test_epochs))
agent.explore=False
FTEST(agent)
FTEST(ra)