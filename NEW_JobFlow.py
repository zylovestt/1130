import numpy as np
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from NEW_Traceback import BEST_PATH
from copy import deepcopy

class JOB:
    def __init__(self,tasks,tasks_col,t):
        self.tasks=tasks
        self.tasks_col=tasks_col
        self.time=t
        self.num=len(tasks)
        # self.real_tasknum=self.tasks['k'].sum()

class PROS:
    def __init__(self,seed,ps,writer:SummaryWriter):
        self.rng=np.random.RandomState(seed)
        self.writer=writer
        self.global_step=0
        self.num=len(ps)
        self.ps=ps
        self.reset()
    
    def set_train_mode(self):
        self.rng=self.train_rng
    
    def set_test_mode(self,seed):
        self.train_rng=self.rng
        self.rng=np.random.RandomState(seed)

    def reset(self):
        for p in self.ps:
            p['alpha']=0.0
            p['beta']=0.0
            p['lx']=0.0
            p['ly']=0.0
        self.time=0.0
    
    def __call__(self,job:JOB,act):
        self.global_step+=1
        assert (np.sum(act,-1)==job.tasks_col['k']).all()
        assert ((~(np.sum(act,0)>0))+[p['k'] for p in self.ps]).all()
        d_l={'o':[],'t':[]}
        timepass=job.time-self.time
        self.time=job.time
        #print('tp ',timepass)
        for index,(p,ac) in enumerate(zip(self.ps,act.T)):
            p['alpha']=max(p['alpha']-timepass,0)
            p['beta']=max(p['beta']-timepass,0)
            # old_loc=np.array([p['lx'],p['ly']])
            if np.sum(ac):
                TU=job.tasks[ac==np.ones_like(ac)]
                TU=[{'r':row['r'],'lx':row['lx'],'ly':row['ly']} for row in TU]
                begin_b=p['beta']-p['alpha']
                best_path=BEST_PATH({'lx':p['lx'],'ly':p['ly'],'b':begin_b},TU,p['v'],p['c'],p['r'])
                best_path.trackback_iter()
                p['alpha']+=best_path.shortest_time_a
                p['beta']+=(best_path.shortest_time-begin_b)
                p['lx']=TU[best_path.best_seq[-1]]['lx']
                p['ly']=TU[best_path.best_seq[-1]]['ly']
                # assert best_path.shortest_time<100
                # print('a ',p['alpha'])
                # print('b ',p['beta'])
                d_l['o'].append(p['beta'])
                d_l['t'].append(best_path.shortest_time-begin_b)
            if self.ps[index]['k']:
                pass
                # self.ps.at[index,'alpha']=p['alpha']
                # self.ps.at[index,'beta']=p['beta']
                # self.ps.at[index,'lx']=p['lx']
                # self.ps.at[index,'ly']=p['ly']
                # self.writer.add_scalar(str(index)+':alpha',p['alpha'],self.global_step)
                # self.writer.add_scalar(str(index)+':beta',p['beta'],self.global_step)
                # self.writer.add_scalar(str(index)+':loc',np.linalg.norm(old_loc-np.array([p['lx'],p['ly']]))/p['v'],self.global_step)
        #     print('i ',index)
        # print('loc:',self.ps['loc'])
        assert len(d_l)
        return d_l

class Pro_Flow:
    def __init__(self,flow_seed,pro_seed,pro_config,num,writer,change=True):
        self.rng=np.random.RandomState(flow_seed)
        self.pro_seed=pro_seed
        self.pro_config=pro_config
        self.num=num
        self.writer=writer
        self.change=change
        # self.fq=lambda s:lambda rng,x:rng.uniform(*s,x)
        self.pros=None
    
    def fq(s):
        def f(rng,x):
            return rng.uniform(*s,x)
        return f

    def set_train_mode(self):
        self.pros.set_train_mode()
        self.rng=self.train_rng
    
    def set_test_mode(self,seed):
        self.pros.set_test_mode(seed) #因为是pros是引用,所以可行
        self.train_rng=self.rng
        self.rng=np.random.RandomState(seed)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.change or self.pros is None:
            rng=self.rng
            n=self.num
            columns=['k','c','r','v','lx','ly'] #k必须放在第一位
            p={key:[0]*n for key in columns}
            # p=pd.DataFrame(np.zeros((self.num,len(columns))),columns=columns)
            RF=lambda x,y:rng.normal(*self.pro_config[x],y)
            # RF=lambda x,y:rng.uniform(*self.pro_config[x],y)
            # num_pro=self.rng.choice(self.num,p=self.pro_config['num_pro'])+1
            num_pro=self.rng.binomial(n-1,self.pro_config['num_pro'])+1
            s=slice(0,num_pro,None)
            p['k'][s]=[1]*num_pro
            p['c'][s]=RF('c',num_pro).tolist()
            p['r'][s]=RF('r',num_pro).tolist()
            p['v'][s]=RF('v',num_pro).tolist()
            # p['lx'][s]=[0.0]*num_pro
            # p['ly'][s]=[0.0]*num_pro
            p=[{k:p[k][i] for k in p} for i in range(n)]
            # num_pro=self.rng.choice(self.num,p=self.pro_config['num_pro'])+1
            # if num_pro<self.num:
            #     p.loc[num_pro:,:]=0.0
            if self.pros is None:
                self.pros=PROS(self.pro_seed,p,self.writer)
            else:
                # self.pros.ps=list(p.T.to_dict().values())
                self.pros.ps=p
            # print('change######################')
        self.pros.reset()
        return self.pros

class Job_Flow:
    def __init__(self,seed,job_config,tasknum,max_length):
        self.rng=np.random.RandomState(seed)
        self.job_config=job_config
        self.tasknum=tasknum
        self.max_length=max_length
        self.delta_time=None
        self.reset()
    
    def set_train_mode(self):
        self.rng=self.train_rng
    
    def set_test_mode(self,seed):
        self.train_rng=self.rng
        self.rng=np.random.RandomState(seed)

    def reset(self):
        rng=self.rng
        tasknum=self.tasknum
        max_length=self.max_length
        job_config=self.job_config
        self.k_num=rng.binomial(tasknum-1,job_config['k'],max_length)
        size=(max_length,tasknum)
        a=np.empty(size)
        a[:]=np.arange(tasknum).reshape(1,-1)
        self.k_mask=(a<=self.k_num.reshape(-1,1)).astype('float')
        self.r=rng.normal(*job_config['r'],size)
        # self.r=rng.uniform(*job_config['r'],size)
        assert (self.r>=0).all()
        loc_mean=rng.uniform(*job_config['loc_mean'],max_length)
        scale=job_config['loc_scale']
        loc_XY=np.empty((max_length,2*tasknum))
        for x,loc in zip(loc_XY,loc_mean):
            x[:]=rng.normal(loc,scale,2*tasknum)
        self.loc_X=loc_XY[:,:tasknum]
        self.loc_Y=loc_XY[:,tasknum:]
        # self.job_time_break=np.array([rng.exponential(x) for x in job_config['time']*max_length])
        self.job_time_break=rng.normal(*job_config['time'],max_length)
        # self.job_time_break=rng.exponential(job_config['time'][0],max_length)
        self.step=0
        self.time=0
    
    def cal_mean_scale(self,epochs):
        l=self.max_length
        self.max_length*=epochs
        rng=deepcopy(self.rng)
        self.reset()
        self.mean={'k':self.k_num.mean(),'r':self.r.mean(),'lx':self.loc_X.mean(),'ly':self.loc_Y.mean(),'t':self.job_time_break.mean()}
        self.scale={'k':self.k_num.std(),'r':self.r.std(),'lx':self.loc_X.std(),'ly':self.loc_Y.std(),'t':self.job_time_break.std()}
        self.max_length=l
        self.rng=rng

    def __iter__(self):
        return self
    
    def __next__(self):
        s=self.step
        k,r,lx,ly,t=self.k_mask[s],self.r[s],self.loc_X[s],self.loc_Y[s],self.job_time_break[s%len(self.job_config['time'])]
        self.step=(self.step+1)%self.max_length
        self.time+=t
        self.delta_time=t
        d_col={'k':k,'r':r,'lx':lx,'ly':ly}
        d=np.array([{k:d_col[k][i] for k in d_col} for i in range(self.tasknum)])
        return JOB(d,d_col,self.time)


if __name__=='__main__':
    jc={'k':0.8,'r':(100,10),'loc_mean':(0,10),'loc_scale':3,'time':[1]}
    jf=Job_Flow(0,jc,3,10)
    jf.reset()
    jf.cal_mean_scale(1000)
    print(jf.mean,jf.scale)
    for _ in range(10):
        print(next(jf))