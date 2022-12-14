import torch
import torch.nn.functional as FU
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import NEW_rl_utils

class PPO:
    def __init__(self,pbeta,epochs,gamma,labda,act_clip_grad,cri_clip_grad,beta,anet,cnet,aoptim,coptim,device,writer:SummaryWriter):
        self.pbeta=pbeta
        self.epochs=epochs
        self.gamma=gamma
        self.labda=labda
        self.act_clip_grad=act_clip_grad
        self.cri_clip_grad=cri_clip_grad
        self.beta=beta
        self.anet=anet
        self.cnet=cnet
        self.aoptim=aoptim
        self.coptim=coptim
        self.device=device
        self.writer=writer
        self.step=0
    
    def take_action(self,state:np.ndarray):
        a=self.anet(torch.tensor(state).to(self.device))[0]
        # print('aaa ',a)
        b=[torch.distributions.Categorical(logits=x).sample().item() for x in a.T]
        act=np.zeros(a.shape,dtype='int32')
        act[b,range(act.shape[1])]=1
        assert (act.sum(axis=0)==1).all(),'act wrong'
        return act
    
    def update(self, transition_dict:dict):
        F=lambda x:torch.tensor(x,dtype=torch.float32).to(self.device)
        states=F(np.concatenate(transition_dict['states'],axis=0))
        next_states=F(np.concatenate(transition_dict['next_states'],axis=0))
        actions=F(np.vstack(transition_dict['actions']).reshape(-1,*transition_dict['actions'][0].shape))
        assert (actions.sum(axis=1)).all(),'actions wrong'
        # print('actions',actions)
        rewards=F(transition_dict['rewards']).reshape(-1,1)
        overs=F(transition_dict['overs']).reshape(-1,1)

        # print(states)
        # print(next_states)
        all_states=torch.concat((states[0:1],next_states),dim=0)
        temp=self.cnet(all_states)
        # print('temp',temp)
        td_target=rewards+self.gamma*temp[1:]*(1-overs)
        td_delta=td_target-temp[:-1]
        # print('td_delta',td_delta)
        dones=transition_dict['dones']
        # print('dones',dones)
        advantage=NEW_rl_utils.compute_advantage_batch(self.gamma, self.labda,td_delta.cpu(),dones).to(self.device)
        old_log_probs_M=(FU.log_softmax(self.anet(states),dim=1)*actions).detach()
        old_log_probs=old_log_probs_M.sum(dim=1).sum(dim=1)

        for _ in range(self.epochs):
            probs=self.anet(states)
            log_probs = (FU.log_softmax(probs,dim=1)*actions).sum(dim=1).sum(dim=1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage.detach()
            kl=((FU.log_softmax(probs,dim=1)-old_log_probs_M)*FU.softmax(probs,dim=1)).sum(dim=1).mean()
            
            actor_loss = -surr1.mean()+self.pbeta*kl
            epo_loss=self.beta*(FU.softmax(probs,dim=1)*FU.log_softmax(probs,dim=1)).sum(1).mean()
            critic_loss=torch.mean(FU.mse_loss(self.cnet(states), td_target.detach()))
            loss=epo_loss+actor_loss+critic_loss
            assert torch.isnan(loss)==0 or torch.isinf(loss)==0,'loss wrong'
            self.anet.zero_grad()
            (actor_loss+epo_loss).backward()
            if not self.act_clip_grad=='max':
                nn_utils.clip_grad_norm_(self.anet.parameters(),self.act_clip_grad)
            self.aoptim.step()
            self.cnet.zero_grad()
            critic_loss.backward()
            if not self.cri_clip_grad=='max':
                nn_utils.clip_grad_norm_(self.cnet.parameters(),self.cri_clip_grad)
            self.coptim.step()

            self.step+=1
            self.writer.add_scalar('critic_loss',critic_loss,self.step)
            self.writer.add_scalar('epo_loss',epo_loss,self.step)
            self.writer.add_scalar('actor_loss',actor_loss,self.step)
            self.writer.add_scalar('total_loss',loss,self.step)
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in self.anet.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
            self.writer.add_scalar('a_grad_l2', grad_means / grad_count, self.step)
            self.writer.add_scalar('a_grad_max', grad_max, self.step)
            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in self.cnet.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
            self.writer.add_scalar('c_grad_l2', grad_means / grad_count, self.step)
            self.writer.add_scalar('c_grad_max', grad_max, self.step)
            probs_new=self.anet(states)
            kl=((FU.log_softmax(probs_new,dim=1)-FU.log_softmax(probs,dim=1))*FU.softmax(probs_new,dim=1)).sum(dim=1).mean().item() # not accurate
            self.writer.add_scalar('KL', kl, self.step)