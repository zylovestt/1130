import itertools
import numpy as np
from copy import deepcopy

class BEST_PATH:
    def __init__(self,begin:dict,tasks:list[dict],v,c,r):
        self.begin=begin
        self.tasks=tasks
        self.v=v
        self.c=[x['r']/c for x in tasks]
        self.r=[x['r']/r for x in tasks]
        self.a=0
        self.b=begin['b']
        assert self.b>=0
        self.cur=0
        self.num_task=len(tasks)
        self.seq=np.arange(self.num_task).tolist()
        self.shortest_time=np.inf
        self.shortest_time_a=np.inf
        self.best_seq=None
        self.generate_loc_array()
    
    # def generate_loc_array(self):
    #     self.loc_array=np.zeros((self.num_task+1,self.num_task+1))
    #     l=[self.begin['loc']]
    #     l.extend([x['loc'] for x in self.tasks])
    #     for i in range(self.num_task+1):
    #         for j in range(i+1,self.num_task+1):
    #             self.loc_array[i,j]=self.loc_array[j,i]=np.linalg.norm(l[i]-l[j])
    #     self.loc_array=self.loc_array.tolist()
    
    def generate_loc_base(self,s):
        x=[self.begin[s]]
        x.extend([u[s] for u in self.tasks])
        x=(np.exp(np.array(x))).reshape(1,-1)
        return np.log(x.T.dot(1/x))**2
    
    def generate_loc_base_cat(self):
        x=[self.begin['lx']]
        x.extend([u['lx'] for u in self.tasks])
        x.append(self.begin['ly'])
        x.extend([u['ly'] for u in self.tasks])
        x=(np.exp(np.array(x))).reshape(1,-1)
        return np.log(x.T.dot(1/x))**2
    
    def generate_loc_array(self):
        if self.num_task>8:
            x=self.generate_loc_base('lx')
            y=self.generate_loc_base('ly')
            self.loc_array=np.sqrt(x+y).tolist()
        else:
            x=self.generate_loc_base_cat()
            self.loc_array=np.sqrt(x[:self.num_task+1,:self.num_task+1]+x[self.num_task+1:,self.num_task+1:])
    
    def cal_loc(self,i,j):
        return self.loc_array[i+1][j+1]

    def swap_seq(self,i,j):
        self.seq[i],self.seq[j]=self.seq[j],self.seq[i]

    def trackback(self):
        if self.cur==self.num_task:
            self.shortest_time=self.b
            self.shortest_time_a=self.a
            self.best_seq=deepcopy(self.seq)
        else:
            for i in range(self.cur,self.num_task):
                self.swap_seq(self.cur,i)
                task=self.seq[self.cur]
                a=self.a
                self.a+=(self.cal_loc(-1 if not self.cur else self.seq[self.cur-1],task)/self.v+self.c[task])
                b=self.b
                self.b=max(self.a,self.b)+self.r[task]
                if self.b<self.shortest_time:
                    self.cur+=1
                    self.trackback()
                    self.cur-=1
                self.a=a
                self.b=b
                self.swap_seq(self.cur,i)
    
    def trackback_iter(self):
        x=[0]*self.num_task
        a=[0]*self.num_task
        b=[0]*self.num_task
        k=0
        x[k]=-1
        a[k]=self.a
        b[k]=self.b
        while(k>-1):
            x[k]+=1
            if x[k]<self.num_task:
                self.swap_seq(k,x[k])
                task=self.seq[k]
                self.a=a[k]+(self.cal_loc(-1 if not k else self.seq[k-1],task)/self.v+self.c[task])
                self.b=max(self.a,b[k])+self.r[task]
                if self.b<self.shortest_time:
                    if k==self.num_task-1:
                        self.shortest_time=self.b
                        self.shortest_time_a=self.a
                        self.best_seq=deepcopy(self.seq)
                    else:
                        k+=1
                        x[k]=k-1
                        a[k]=self.a
                        b[k]=self.b
                else:
                    self.swap_seq(k,x[k])
            else:
                k-=1
                if k>-1 and x[k]<self.num_task:
                    self.swap_seq(k,x[k])
    
    def iter_element(self,seq):
        a=self.a
        b=self.b
        for k,task in enumerate(seq):
            a+=(self.cal_loc(-1 if not k else seq[k-1],task)/self.v+self.c[task])
            b=max(a,b)+self.r[task]
        return a,b
    
    def pure_iter(self):
        for seq in itertools.permutations(self.seq,len(self.seq)):
            a,b=self.iter_element(seq)
            if b<self.shortest_time:
                self.shortest_time=b
                self.shortest_time_a=a
                self.best_seq=list(deepcopy(seq))

if __name__=='__main__':
    import time
    begin={'lx':0,'ly':0,'b':0}
    tasks=[{'lx':0,'ly':1,'r':2},{'lx':1,'ly':0,'r':1},{'lx':3,'ly':4,'r':2},{'lx':0,'ly':0,'r':1}]
    start=time.time()
    tr=BEST_PATH(begin,tasks,1,1,1)
    tr.trackback()
    print(tr.shortest_time,tr.shortest_time_a,tr.best_seq)
    print(tr.seq)
    print(time.time()-start)

    start=time.time()
    tr=BEST_PATH(begin,tasks,1,1,1)
    tr.trackback_iter()
    print(tr.shortest_time,tr.shortest_time_a,tr.best_seq)
    print(tr.seq)
    print(time.time()-start)

    start=time.time()
    tr=BEST_PATH(begin,tasks,1,1,1)
    tr.pure_iter()
    print(tr.shortest_time,tr.shortest_time_a,tr.best_seq)
    print(tr.seq)
    print(time.time()-start)
    print(np.array(tr.loc_array))