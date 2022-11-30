import torch
from NEW_AC import AC
from NEW_PUBLIC_ENV import RandomAgent,train_on_policy_agent,model_test,env,anet,cnet,device,writer

aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-4,eps=1e-8) # lr=1e-4
coptim=torch.optim.NAdam(cnet.parameters(),lr=1e-3,eps=1e-8) # lr=1e-2
# gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device
# 0.96,0.95,1e-1,1e-1,1e-3
agent=AC(0.95,0.95,1e-1,1e-1,1,anet,cnet,aoptim,coptim,device,writer)
test_epochs=50
train_on_policy_agent(0,env,agent,1000,5,writer,100,test_epochs)
ra=RandomAgent(9,env.pros.num,env.jf.tasknum)
FTEST=lambda x:print(model_test(0,env,x,test_epochs))
agent.explore=False
FTEST(agent)
FTEST(ra)