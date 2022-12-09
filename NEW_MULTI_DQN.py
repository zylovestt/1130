import torch
import torch.nn.functional as FU
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import NEW_rl_utils
from NEW_STATE import fstate
from copy import deepcopy
from NEW_TD3 import onehot_from_logits

# def onehot_from_logits(logits, eps):
#     ''' 生成最优动作的独热(one-hot)形式 '''
#     logits+=torch.randn(size=logits.shape,device=logits.device)*1e-1
#     hhh=(logits == logits.max(-1, keepdim=True)[0]).float()
#     while not (hhh.sum(-1)<1.5).all():
#         logits+=torch.randn(size=logits.shape,device=logits.device)*1e-1
#         hhh=(logits == logits.max(-1, keepdim=True)[0]).float()
#         print('same')
#     assert (hhh.sum(-1)<1.001).all()
#     return hhh

class MULTI_DQN:
    def __init__(self,gamma,qnet,qoptim,tau,device):
        self.gamma=gamma
        self.actor=qnet
        self.target_actor=deepcopy(self.actor)
        self.actor_optimizer=qoptim
        self.critic_criterion = torch.nn.MSELoss()
        self.device=device
        self.explore=True
        self.tau=tau
        self.task_num=self.actor.task_num
        self.pro_num=self.actor.pro_num
        self.critic1=self.critic2=self.target_critic1=self.target_critic2=None
        self.critic_optimizer1=self.critic_optimizer2=None
    
    def take_action(self,state:np.ndarray):
        F=lambda x:torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
        q_value=self.actor(F(state).unsqueeze(0))[0]
        act=onehot_from_logits(q_value,None).detach().cpu().numpy()
        assert (act.sum(axis=-1)==1).all(),'act wrong'
        return act
    
    def update(self, transition_dict:dict):
        states,next_states,actions,rewards=(transition_dict['states'],transition_dict['next_states'],
                                            transition_dict['actions'],transition_dict['rewards'])
        with torch.no_grad():
            q_target=(rewards+self.gamma*self.target_actor(next_states).max(axis=-1)[0])
        # loss=self.critic_criterion((self.actor(states)*(actions.reshape(-1,8,8))).sum(dim=-1),q_target)
        loss=FU.mse_loss((self.actor(states)*(actions.reshape(-1,self.task_num,self.pro_num))).sum(dim=-1),q_target)
        # print('loss',loss)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.update_all_targets()
    
    def update_all_targets(self):
        self.soft_update(self.actor, self.target_actor)
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        