import numpy as np
from torch.utils.tensorboard import SummaryWriter
from NEW_JobFlow import Pro_Flow,Job_Flow

class NEW_ENV:
    def __init__(self,pf:Pro_Flow,jf:Job_Flow,maxsteps,writer:SummaryWriter):
        self.pf=pf
        self.jf=jf
        self.md=0
        self.sd=1
        self.maxsteps=maxsteps
        self.global_step=0
        self.writer=writer
        self.D=[]
        self.job_list=[]
        state=self.reset()
        self.state_size=state.size
        self.train=True
    
    def set_train_mode(self):
        self.train=True
        self.pf.set_train_mode()
        self.jf.set_train_mode()
        self.D=self.old_D

    def set_test_mode(self,seed):
        self.train=False
        self.pf.set_test_mode(seed)
        self.jf.set_test_mode(seed)
        self.old_D=self.D
        
    def normalize(self,epochs):
        assert epochs,'epochs is zero'
        # self.mq=self.md=0
        # self.sq=self.sd=1
        self.D.clear()
        ra=RandomAgent(0,self.pros.num,self.jf.tasknum)
        for _ in range(epochs):
            state=self.reset()
            done=0
            while not done:
                act=ra.take_action(state)
                state,_,done,_=self.step(act)
        D=np.array(self.D)
        self.md=np.mean(D)
        self.sd=np.std(D)
        # self.md=(np.max(D)+np.min(D))/2
        # self.sd=np.max(D)
        assert self.sd,'std is zero'
        print('MD:',self.md,'SD:',self.sd,'DIV:',self.sd/self.md)
        self.D.clear()
        self.global_step,self.pros.global_step=0,0

    def send(self,maddpg=False):
        pros,job=self.pros,self.job
        # s_pro=pros.ps[['k','c','r','v','alpha','beta','lx','ly'] if self.pf.change else ['k','alpha','beta','lx','ly']].values
        # s_pro=pros.ps[['k','c','r','v','alpha','beta','lx','ly']].values
        s_pro=np.array([list(t.values()) for t in pros.ps])
        s_job=np.array([[self.stepnum,self.jf.delta_time]])
        # s_job=np.array([[self.jf.delta_time]])
        # s_job=np.array([[job.time]])
        # jt=deepcopy(job.tasks)
        # ind=['r','lx','ly']
        # jt[ind]=(jt[ind]-np.array([[self.jf.mean[key] for key in ind]]))/np.array([[self.jf.scale[key] for key in ind]])
        # s_task=np.concatenate([s_job,jt.values.T.reshape(1,-1)],axis=1)
        # s_task=np.concatenate([s_job,job.tasks.values.T.reshape(1,-1)],axis=1)
        s_task=np.array(list(job.tasks_col.values()))
        result=np.concatenate([s_pro.T.reshape(1,-1),s_job,s_task.reshape(1,-1)],axis=1).astype(np.float32)
        if maddpg:
            return [result.reshape(-1)]*self.jf.tasknum
        return result.reshape(-1)

    def newjob(self):
        self.job=next(self.jf)
        # self.kmax=max(self.job.tasks_col['k'])
        # assert self.kmax>0
        # self.l=0

    def reset(self,maddpg=False):
        self.pros=next(self.pf)
        # print(self.pros.ps)
        self.jf.reset()
        self.newjob()
        self.stepnum=0
        return self.send(maddpg)
    
    def step(self,act,maddpg=False): # 输入的act是原始的
        if maddpg:
            act=np.vstack(act)
            act=(act==act.max(-1,keepdims=True)).astype('int')
            # print(act)
        # print('1 ',mask)
        # print('2 ',act)
        act=act.copy()
        # act=(act==act.max(-1,keepdims=True)).astype('int')
        act[self.job.tasks_col['k']==0]=0
        done=0
        # print('3 ',mask)
        if self.train:
            D=np.max(self.pros(self.job,act)['o'])
            self.D.append(D)
            D=(D-self.md)/self.sd
            reward=-D
            # self.writer.add_scalar('R',reward,self.global_step)
        else:
            D=np.max(self.pros(self.job,act)['o'])
            self.D.append(D)
            D=(D-self.md)/self.sd
            reward=-D
            # self.writer.add_scalar('test_time_delay',reward,self.global_step)
        self.global_step+=1
        self.stepnum+=1
        if self.stepnum>=self.maxsteps:
            assert self.stepnum==self.maxsteps
            # print('done')
            done=1
        else:
            self.newjob()
        send=self.send(maddpg)
        if maddpg:
            return send,[reward]*self.jf.tasknum,[done]*self.jf.tasknum,None
        return send,reward,done,0
    
class RandomAgent:
    def __init__(self,seed,pn,tn):
        self.rng=np.random.RandomState(seed)
        self.pn=pn
        self.tn=tn

    def take_action(self,state):
        act=np.zeros((self.tn,self.pn),dtype=np.int32)
        r=self.rng.choice(int(state[:self.pn].sum()),self.tn)
        act[range(self.tn),r]=1
        return act

# if __name__=='__main__':
#     writer=SummaryWriter(comment='NEW_ENV')
#     print('test1')
#     fq=lambda s:lambda rng,x:rng.uniform(*s,x)
#     p=pd.DataFrame([[1,1,1,np.array([0,0]),(1,3)],
#                     [11,1,11,np.array([1,0]),(6,7)],
#                     [100,10,120,np.array([0,2]),(5,6)]],columns=['c','r','v','loc','q'],index=['P1','P2','P3'])
#     for i,_ in p.iterrows():
#         p.at[i,'q']=fq(p.loc[i,'q'])
#     p=PROS(0,p,writer)
#     # p.ps['alpha']=[1,4,2]
#     # p.ps['beta']=[2,1,3]
#     # p.time=3
#     t=pd.DataFrame([[10,20,1],
#                     [20,10,1],],columns=['c','r','k'])
#     t=JOB(t,np.array([3,0]),5,np.array([[2,2],[0,0],[0,0]]))
#     # print(p(t))
#     # print(p(t))

#     print('test2')
#     pronum=3
#     m=pd.DataFrame([[(1,10),(2,20),(1,4)],
#                     [(1,10),(2,20),(1,4)]],columns=['c','r','k'],index=['T1','T2'])
#     d={'time':(1,5),
#        'loc':(1,20,2),
#        'task':m}
#     jp=Job_Flow(10,d,pronum)
#     i=0
#     for j in jp:
#         #print(j.tasks)
#         i+=1
#         if i==1:
#             break
    
#     print('test3')
#     env=NEW_ENV(p,jp,1,1,0,50,writer)
#     print(env.reset())

#     print('test4')
#     env.normalize(10)
#     # print('nor over')
#     print(env.mq,env.sq,env.md,env.sd)
#     ra=RandomAgent(0,pronum,env.jf.tasknum)
#     from NEW_NET import ActNet,CriticNet
#     import torch
#     torch.manual_seed(2)
#     state=env.reset()
#     anet=ActNet(3,100,3,100,state.size,env.pros.num,env.jf.tasknum)
#     print(anet(torch.tensor(np.concatenate([state,state],0))))
#     cnet=CriticNet(3,100,state.size)
#     print(cnet(torch.tensor(np.concatenate([state,state],0))))

#     print('test5')
#     from NEW_AC import AC
#     aoptim=torch.optim.NAdam(anet.parameters(),lr=1e-4,eps=1e-8)
#     coptim=torch.optim.NAdam(cnet.parameters(),lr=1e-2,eps=1e-8)
#     # gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device
#     agent=AC(0.98,0.95,'max','max',1e-4,anet,cnet,aoptim,coptim,'cpu')
#     print(agent.take_action(state))

#     print('test6')
#     import NEW_rl_utils
#     NEW_rl_utils.train_on_policy_agent(9,env, agent, 1000,10)

#     from NEW_TEST import model_test
#     ra=RandomAgent(9,pronum,env.jf.tasknum)
#     FTEST=lambda x:print(model_test(0,env,x,20))
#     FTEST(agent)
#     FTEST(ra)