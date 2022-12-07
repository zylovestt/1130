from copy import deepcopy
import torch
import numpy as np
import torch.nn.functional as FU
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter

def onehot_from_logits(logits, eps):
    ''' 生成最优动作的独热(one-hot)形式 '''
    logits+=torch.randn(size=logits.shape,device=logits.device)*1e-1
    hhh=(logits == logits.max(-1, keepdim=True)[0]).float()
    while not (hhh.sum(-1)<1.5).all():
        logits+=torch.randn(size=logits.shape,device=logits.device)*1e-1
        hhh=(logits == logits.max(-1, keepdim=True)[0]).float()
        print('same')
    assert (hhh.sum(-1)<1.001).all()
    return hhh

class MULTI_SAC:
    ''' 处理离散动作的SAC算法 '''    
    def __init__(self, anet:torch.nn.Module,qnet1:torch.nn.Module,qnet2:torch.nn.Module,aoptim,qoptim1,qoptim2, tau, gamma, device,target_entropy,alpha_lr):
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
        self.critic_criterion = torch.nn.MSELoss()
        self.c_epochs=3
        self.update_cycles=2
        self.explore=True
        self.eps=0.0
        self.tem=1
        self.num_update=0
        self.target_actor.requires_grad_(False)
        self.target_critic1.requires_grad_(False)
        self.target_critic2.requires_grad_(False)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小

    def take_action(self,state:np.ndarray):
        # a=self.actor(fstate(lambda x:torch.tensor(x,dtype=torch.float32).to(self.device),state))[0]
        F=lambda x:torch.tensor(np.array(x),dtype=torch.float32).to(self.device)
        a=self.actor(F(state).unsqueeze(0))[0]
        # print(a)
        if self.explore:
            b=torch.distributions.Categorical(logits=a).sample().detach().cpu().numpy()
            act=np.zeros(a.shape,dtype='int32')
            act[range(act.shape[0]),b]=1
        else:
            # act = (a == a.max(-1, keepdim=True)[0]).detach().cpu().numpy().astype('int32')
            act=onehot_from_logits(a,None).detach().cpu().numpy()
        assert (act.sum(axis=-1)==1).all(),'act wrong'
        return act

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = FU.softmax(self.actor(next_states),dim=-1)
        # print(next_probs[0][0])
        next_log_probs = FU.log_softmax(self.actor(next_states),dim=-1)
        entropy = -torch.sum(next_probs * next_log_probs, dim=-1, keepdim=False)
        q1_value = self.target_critic1(next_states)
        q2_value = self.target_critic2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=-1,
                               keepdim=False)
        next_value = min_qvalue + self.log_alpha.exp()* entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        # print(td_target[0])
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states,next_states,actions,rewards=(transition_dict['states'],transition_dict['next_states'],
                                            transition_dict['actions'],transition_dict['rewards'])

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, 0)
        critic_1_q_values = (self.critic1(states)*actions.reshape(-1,8,8)).sum(dim=-1)
        critic_1_loss = torch.mean(FU.mse_loss(critic_1_q_values, td_target.detach()))
        # print('c1',critic_1_loss)

        critic_2_q_values = (self.critic2(states)*actions.reshape(-1,8,8)).sum(dim=-1)
        critic_2_loss = torch.mean(FU.mse_loss(critic_2_q_values, td_target.detach()))
        # print('c2',critic_2_loss)

        self.critic_optimizer1.zero_grad()
        critic_1_loss.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_2_loss.backward()
        self.critic_optimizer2.step()

        # 更新策略网络
        probs = FU.softmax(self.actor(states),dim=-1)
        log_probs = FU.log_softmax(self.actor(states),dim=-1)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=False)  #
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=-1,
                               keepdim=False)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        # print('al',actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()

        #更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()) #change
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        # print('entropy',entropy.max())
        # print('alpha_loss',alpha_loss)
        # print('alpha',self.log_alpha.exp())
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)