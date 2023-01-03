from copy import deepcopy
import torch
import numpy as np
import torch.nn.functional as FU
import torch.nn.utils as nn_utils
# from torch.utils.tensorboard import SummaryWriter

# def onehot_from_logits(logits, eps):
#     ''' 生成最优动作的独热(one-hot)形式 '''
#     argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
#     rand_acs = torch.autograd.Variable(torch.eye(logits.shape[-1])[[
#         np.random.choice(range(logits.shape[-1]), size=logits.shape[:-1])]],requires_grad=False).to(logits.device)
#     return torch.stack([torch.stack([argmax_acs[i,j] if r > eps else rand_acs[i,j] for j,r in enumerate(r_i)]) for i,r_i in enumerate(torch.rand(logits.shape[:-1]))])

def onehot_from_logits(logits, eps, beta1=1e-1,beta2=1e-1):
    ''' 生成最优动作的独热(one-hot)形式 '''
    r=torch.randn(size=logits.shape,device=logits.device)*beta1
    logits+=r
    hhh=(logits == logits.max(-1, keepdim=True)[0]).float()
    while not (hhh.sum(-1)<1.5).all():
        r=torch.randn(size=logits.shape,device=logits.device)*beta2
        logits+=r
        hhh=(logits == logits.max(-1, keepdim=True)[0]).float()
        print('same')
        # index=hhh.sum(-1).reshape(-1).argmax()
        # act_size=logits.shape[-2]*logits.shape[-1]
        # u=logits[index//act_size,(index%act_size)//logits.shape[]]
    assert (hhh.sum(-1)<1.001).all()
    return hhh

def onehot_from_logits_quick(logits, eps, beta1=1e-1):
    ''' 生成最优动作的独热(one-hot)形式 '''
    r=torch.randn(size=logits.shape,device=logits.device)*beta1
    logits+=r
    hhh=(logits == logits.max(-1, keepdim=True)[0]).float()
    return hhh
    
def sample_gumbel(shape, eps=1e-10, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return FU.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature,eps):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y,eps)
    y = (y_hard.to(logits.device) - y).detach() + y
    # y = (y_hard.to(logits.device) + y).detach() - y
    return y

class TD3:
    def __init__(self, anet:torch.nn.Module,qnet1:torch.nn.Module,qnet2:torch.nn.Module,aoptim,qoptim1,qoptim2, tau, gamma, device,writer,clip_grad,conn,curs,date_time):
        self.name='td3'
        self.actor = anet
        self.target_actor=deepcopy(anet)
        self.critic1 = qnet1
        self.critic2 = qnet2
        self.target_critic1 = deepcopy(qnet1)
        self.target_critic2 = deepcopy(qnet2)
        self.actor_optimizer = aoptim
        self.critic_optimizer1 = qoptim1
        self.critic_optimizer2 = qoptim2
        self.gamma=gamma
        self.tau=tau
        self.device=device
        self.writer=writer
        self.critic_criterion = torch.nn.MSELoss()
        self.clip_grad=clip_grad
        self.c_epochs=3
        self.a_epochs=1
        self.update_cycles=2
        self.explore=True
        self.eps=0.0
        self.tem=1
        self.beta1=1e-1
        self.beta2=1e-1
        self.num_update=0
        self.conn=conn
        self.curs=curs
        self.date_time=date_time
        self.target_actor.requires_grad_(False)
        self.target_critic1.requires_grad_(False)
        self.target_critic2.requires_grad_(False)
    
    def insert_data(self,step,name,value):
        self.curs.execute("insert into recordvalue values('%s','td3','%s',%d,%f)"%(self.date_time,name,step,value))
    
    def take_action(self,state):
        # a=(self.actor(fstate(lambda x:torch.tensor(x,dtype=torch.float32).to(self.device),state)))
        F=lambda x:torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
        with torch.no_grad():
            a=self.actor(F(state).unsqueeze(0))
        if self.explore:
            # a=gumbel_softmax(a,self.tem,self.eps)[0]
            a=FU.gumbel_softmax(a,tau=self.tem,dim=-1,hard=True)[0]
            # a=FU.gumbel_softmax(a,tau=self.tem,dim=-1,hard=False)[0]
            # b=torch.distributions.Categorical(logits=a[0]).sample().detach().cpu().numpy()
            # act=np.zeros(a.shape[1:],dtype='int32')
            # act[range(act.shape[0]),b]=1
            # return act
        else:
            a=onehot_from_logits(a,self.eps,self.beta1,self.beta2)[0]
        return a.detach().cpu().numpy()
        # return (a.detach().cpu().numpy()[0]>0.99).astype('int')
    
    # def take_action(self,state:np.ndarray):
    #     # a=self.actor(fstate(lambda x:torch.tensor(x,dtype=torch.float32).to(self.device),state))[0]
    #     F=lambda x:torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
    #     a=self.actor(F(state).unsqueeze(0))[0]
    #     # print(a)
    #     if self.explore:
    #         b=torch.distributions.Categorical(logits=a).sample().detach().cpu().numpy()
    #         act=np.zeros(a.shape,dtype='int32')
    #         act[range(act.shape[0]),b]=1
    #     else:
    #         # act = (a == a.max(-1, keepdim=True)[0]).detach().cpu().numpy().astype('int32')
    #         act=onehot_from_logits(a,None).detach().cpu().numpy()
    #     assert (act.sum(axis=-1)==1).all(),'act wrong'
    #     return act
    
    def update(self,transition_dict:dict):
        self.num_update+=1
        if (self.num_update%500)==0:
            # if self.eps>0.01:
            #     self.eps*=0.96
            # if self.tem>0.1:
            #     self.tem*=0.96
            upbound=np.exp(-self.num_update*1e-5)
            self.tem=max(0.5,upbound)
            self.beta1=max(1e-3,upbound*1e-1)
            self.beta2=max(1e-3,upbound*1e-1)

        # F=lambda x:torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
        # # states=fstate(lambda x:F(np.concatenate(x,axis=0)),transition_dict['states'])
        # # next_states=fstate(lambda x:F(np.concatenate(x,axis=0)),transition_dict['next_states'])
        # states=F(transition_dict['states'])
        # next_states=F(transition_dict['next_states'])
        # actions=F(np.vstack(transition_dict['actions']).reshape(-1,*transition_dict['actions'][0].shape))
        # assert (actions.sum(axis=-1)).all(),'actions wrong'
        # # print('actions',actions)
        # rewards=F(transition_dict['rewards']).reshape(-1,1)

        states,next_states,actions,rewards=(transition_dict['states'],transition_dict['next_states'],
                                            transition_dict['actions'],transition_dict['rewards'])

        
        target_act=onehot_from_logits_quick(self.target_actor(next_states),self.eps,self.beta1) #@@@@@@@@@@@@@@@@@@@
        # target_act=FU.gumbel_softmax(self.target_actor(next_states),tau=self.tem,dim=-1,hard=False) #@@@@@@@@@@@@@@@@@@@
        # target_act=gumbel_softmax(self.target_actor(next_states),self.tem,self.eps) #@@@@@@@@@@@@@@@@@@@
        target_critic_input = torch.cat((next_states,target_act.reshape(target_act.shape[0],-1)), dim=-1)
        target_critic_value = rewards + self.gamma * torch.min(
            self.target_critic1(target_critic_input),self.target_critic2(target_critic_input)) #change
        # critic_input = torch.cat((states,actions.reshape(actions.shape[0],-1)),dim=1)
        critic_input = torch.cat((states,actions),dim=-1)
        for i in range(self.c_epochs):
            critic_value1 = self.critic1(critic_input)
            critic_value2 = self.critic2(critic_input)
            critic_loss1 = self.critic_criterion(critic_value1,target_critic_value.detach())
            self.critic_optimizer1.zero_grad()
            self.critic_optimizer2.zero_grad()
            critic_loss1.backward()
            if not self.clip_grad=='max':
                    nn_utils.clip_grad_norm_(self.critic1.parameters(),self.clip_grad)
            self.critic_optimizer1.step()
            critic_loss2 = self.critic_criterion(critic_value2,target_critic_value.detach())
            critic_loss2.backward()
            if not self.clip_grad=='max':
                    nn_utils.clip_grad_norm_(self.critic2.parameters(),self.clip_grad)
            self.critic_optimizer2.step()
            self.insert_data(self.c_epochs*self.num_update+i,'critic_loss1',critic_loss1)
            self.insert_data(self.c_epochs*self.num_update+i,'critic_loss2',critic_loss2)
            # self.conn.commit()
            # self.writer.add_scalar('critic_loss1',critic_loss1,self.c_epochs*self.num_update+i)
            # self.writer.add_scalar('critic_loss2',critic_loss2,self.c_epochs*self.num_update+i)

        if (self.num_update%self.update_cycles)==0:
            for i in range(self.a_epochs):
                self.actor_optimizer.zero_grad()
                actor_out = self.actor(states)
                # act_vf_in = gumbel_softmax(actor_out,self.tem,self.eps)
                act_vf_in=FU.gumbel_softmax(actor_out,tau=self.tem,dim=-1,hard=True)
                # act_vf_in=FU.gumbel_softmax(actor_out,tau=self.tem,dim=-1,hard=False)
                vf_in = torch.cat((states, act_vf_in.reshape(act_vf_in.shape[0],-1)), dim=1)

                
                self.critic1.requires_grad_(False)
                self.critic2.requires_grad_(False)
                actor_loss = -torch.min(self.critic1(vf_in),self.critic2(vf_in)).mean() #ffffffffffffffff
                # actor_loss = -self.critic1(vf_in).mean() #ffffffffffffffff
                self.critic1.requires_grad_(True)
                self.critic2.requires_grad_(True)
                actor_norm = (((actor_out>-1e7).float()*actor_out)**2).mean()
                # actor_loss += actor_norm*1e-3
                epo_loss=(FU.softmax(actor_out,dim=-1)*FU.log_softmax(actor_out,dim=-1)).sum(dim=-1).mean()
                # (actor_loss+epo_loss*1e-1+actor_norm*1e-3).backward()
                (actor_loss+actor_norm*1e-3).backward()
                # actor_loss.backward()
                if not self.clip_grad=='max':
                        nn_utils.clip_grad_norm_(self.actor.parameters(),self.clip_grad)
                self.actor_optimizer.step()
                step=self.a_epochs*self.num_update//self.update_cycles+i
                self.insert_data(step,'actor_loss',actor_loss)
                self.insert_data(step,'actor_norm',actor_norm)
                self.insert_data(step,'epo_loss',epo_loss)
                # self.conn.commit()
                # self.writer.add_scalar('actor_loss',actor_loss,self.num_update//self.update_cycles)
                # self.writer.add_scalar('actor_norm',actor_norm,self.num_update//self.update_cycles)
                # self.writer.add_scalar('epo_loss',epo_loss,self.num_update//self.update_cycles)
            self.update_all_targets()

    def update_all_targets(self):
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self,path):
        torch.save(self.actor,path)
    
    def load_model(self,path):
        self.actor=torch.load(path)

class DDPG:
    def __init__(self, anet:torch.nn.Module,qnet1:torch.nn.Module,qnet2:torch.nn.Module,aoptim,qoptim1,qoptim2, tau, gamma, device,writer,clip_grad,conn,curs,date_time):
        self.name='ddpg'
        self.actor = anet
        self.target_actor=deepcopy(anet)
        self.critic1 = qnet1
        self.critic2 = qnet2
        self.target_critic1 = deepcopy(qnet1)
        self.target_critic2 = deepcopy(qnet2)
        self.actor_optimizer = aoptim
        self.critic_optimizer1 = qoptim1
        self.critic_optimizer2 = qoptim2
        self.gamma=gamma
        self.tau=tau
        self.device=device
        self.writer=writer
        self.critic_criterion = torch.nn.MSELoss()
        self.clip_grad=clip_grad
        self.c_epochs=3
        self.a_epochs=3
        self.update_cycles=1
        self.explore=True
        self.eps=0.0
        self.tem=1
        self.num_update=0
        self.conn=conn
        self.curs=curs
        self.date_time=date_time
        self.target_actor.requires_grad_(False)
        self.target_critic1.requires_grad_(False)
        self.target_critic2.requires_grad_(False)
    
    def insert_data(self,step,name,value):
        self.curs.execute("insert into recordvalue values('%s','ddpg','%s',%d,%f)"%(self.date_time,name,step,value))
    
    def take_action(self,state):
        # a=(self.actor(fstate(lambda x:torch.tensor(x,dtype=torch.float32).to(self.device),state)))
        F=lambda x:torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
        with torch.no_grad():
            a=self.actor(F(state).unsqueeze(0))
        if self.explore:
            # a=gumbel_softmax(a,self.tem,self.eps)[0]
            a=FU.gumbel_softmax(a,tau=self.tem,dim=-1,hard=True)[0]
        else:
            a=onehot_from_logits(a,self.eps,0,1e-1)[0]
        return a.detach().cpu().numpy()
    
    def update(self,transition_dict:dict):
        self.num_update+=1
        if (self.num_update%500)==0:
            self.tem=max(0.5,np.exp(-self.num_update*1e-5))
        states,next_states,actions,rewards=(transition_dict['states'],transition_dict['next_states'],
                                            transition_dict['actions'],transition_dict['rewards'])

        
        target_act=onehot_from_logits(self.target_actor(next_states),self.eps,0,1e-1) #@@@@@@@@@@@@@@@@@@@
        # target_act=FU.gumbel_softmax(self.target_actor(next_states),tau=self.tem,dim=-1,hard=False) #@@@@@@@@@@@@@@@@@@@
        # target_act=gumbel_softmax(self.target_actor(next_states),self.tem,self.eps) #@@@@@@@@@@@@@@@@@@@
        target_critic_input = torch.cat((next_states,target_act.reshape(target_act.shape[0],-1)), dim=-1)
        target_critic_value = rewards + self.gamma * self.target_critic1(target_critic_input) #change
        # critic_input = torch.cat((states,actions.reshape(actions.shape[0],-1)),dim=1)
        critic_input = torch.cat((states,actions),dim=-1)
        for i in range(self.c_epochs):
            critic_value1 = self.critic1(critic_input)
            # critic_value2 = self.critic2(critic_input)
            critic_loss1 = self.critic_criterion(critic_value1,target_critic_value.detach())
            self.critic_optimizer1.zero_grad()
            # self.critic_optimizer2.zero_grad()
            critic_loss1.backward()
            if not self.clip_grad=='max':
                    nn_utils.clip_grad_norm_(self.critic1.parameters(),self.clip_grad)
            self.critic_optimizer1.step()
            self.insert_data(self.c_epochs*self.num_update+i,'critic_loss1',critic_loss1)

        if (self.num_update%self.update_cycles)==0:
            for i in range(self.a_epochs):
                
                actor_out = self.actor(states)
                # act_vf_in = gumbel_softmax(actor_out,self.tem,self.eps)
                act_vf_in=FU.gumbel_softmax(actor_out,tau=self.tem,dim=-1,hard=True)
                # act_vf_in=FU.gumbel_softmax(actor_out,tau=self.tem,dim=-1,hard=False)
                vf_in = torch.cat((states, act_vf_in.reshape(act_vf_in.shape[0],-1)), dim=1)
                self.critic1.requires_grad_(False)
                # self.critic2.requires_grad_(False)
                actor_loss = -self.critic1(vf_in).mean() #ffffffffffffffff
                # actor_loss = -self.critic1(vf_in).mean() #ffffffffffffffff
                self.critic1.requires_grad_(True)
                # self.critic2.requires_grad_(True)
                actor_norm = (((actor_out>-1e7).float()*actor_out)**2).mean()
                # actor_norm = (actor_out**2).mean()
                # actor_loss += actor_norm*1e-3
                epo_loss=(FU.softmax(actor_out,dim=-1)*FU.log_softmax(actor_out,dim=-1)).sum(dim=-1).mean()
                self.actor_optimizer.zero_grad()
                # (actor_loss+epo_loss*2e-1).backward()
                (actor_loss+actor_norm*1e-3).backward()
                # actor_loss.backward()
                if not self.clip_grad=='max':
                        nn_utils.clip_grad_norm_(self.actor.parameters(),self.clip_grad)
                self.actor_optimizer.step()
                step=self.a_epochs*self.num_update//self.update_cycles+i
                self.insert_data(step,'actor_loss',actor_loss)
                self.insert_data(step,'actor_norm',actor_norm)
                self.insert_data(step,'epo_loss',epo_loss)
            self.update_all_targets()

    def update_all_targets(self):
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic1, self.target_critic1)
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self,path):
        torch.save(self.actor,path)
    
    def load_model(self,path):
        self.actor=torch.load(path)