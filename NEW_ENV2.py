import numpy as np
from torch.utils.tensorboard import SummaryWriter
from NEW_JobFlow import Pro_Flow,Job_Flow_Change

class NEW_ENV:
    def __init__(self,pf:Pro_Flow,jf:Job_Flow_Change,maxsteps,writer:SummaryWriter):
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
        print(state)
        self.action_size=pf.num*jf.tasknum
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
        # s_pro=np.array([list(t.values() if t=='k' else t.values()/100) for t in pros.ps])
        s_pro=np.array([list(t.values()) for t in pros.ps])
        # s_pro[:,1:]/=10
        s_pro[:,-2:]=np.maximum(s_pro[:,-2:]-self.jf.delta_time,0)
        s_pro[:,-2:]/=100
        # s_job=np.array([[self.stepnum,self.jf.delta_time]])
        # s_job=np.array([[self.jf.delta_time]])

        # s_job=np.array([[job.time]])
        # jt=deepcopy(job.tasks)
        # ind=['r','lx','ly']
        # jt[ind]=(jt[ind]-np.array([[self.jf.mean[key] for key in ind]]))/np.array([[self.jf.scale[key] for key in ind]])
        # s_task=np.concatenate([s_job,jt.values.T.reshape(1,-1)],axis=1)
        # s_task=np.concatenate([s_job,job.tasks.values.T.reshape(1,-1)],axis=1)

        s_task=np.array(list(job.tasks_col.values()))
        # s_task[1:]/=100
        # s_task=np.array([list(t.values() if t=='k' else t.values()/100) for t in job.tasks])
        # result=np.concatenate([s_pro.T.reshape(1,-1),s_job,s_task.reshape(1,-1)],axis=1).astype(np.float32)
        result=np.concatenate([s_pro.T.reshape(1,-1),s_task.reshape(1,-1)],axis=1).astype(np.float32)
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
            # act=(act==act.max(-1,keepdims=True)).astype('int')
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
    
    # def take_action(self,state):
    #     act=np.zeros((self.tn,self.pn),dtype=np.int32)
    #     r=self.rng.choice(self.pn,size=self.tn,p=state[:self.pn]/state[:self.pn].sum())
    #     act[range(self.tn),r]=1
    #     return act

if __name__=='__main__':
    from NEW_JobFlow import Job_Flow
    device='cuda'
    writer=None
    pro_config={'c':(1,0),'r':(1,0),'v':(1,0)}
    pro_num=2
    pro_config['num_pro']=1
    PF=Pro_Flow(1,0,pro_config,pro_num,writer,True)
    jc={'k':1,'r':(1,0),'loc_mean':(1,1),'loc_scale':0,'time':(1,0)}
    tasknum=2
    env_steps=10
    JF=Job_Flow(0,jc,tasknum,env_steps)
    env=NEW_ENV(PF,JF,env_steps,writer)
    state=env.reset()
    act=np.array([[1,0],[0,1]])
    print(env.step(act))
    print(env.pros.ps)
    act=np.array([[1,0],[1,0]])
    print(env.step(act))
    print(env.pros.ps)