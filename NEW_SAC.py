import numpy as np
import torch
import torch.nn.functional as FU
from copy import deepcopy

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, anet,qnet1,qnet2,aoptim,qoptim1,qoptim2,alpha_lr, target_entropy, tau, gamma, device, writer):
        # 策略网络
        self.actor = anet
        # 第一个Q网络
        self.critic_1 = qnet1
        # 第二个Q网络
        self.critic_2 = qnet2
        self.target_critic_1 = deepcopy(qnet1)
        self.target_critic_2 = deepcopy(qnet2)
        # 令目标Q网络的初始参数和Q网络一样
        
        self.actor_optimizer = aoptim
        self.critic_1_optimizer = qoptim1
        self.critic_2_optimizer = qoptim2
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float) #should change
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.writer=writer
        self.step=0

    def take_action(self, state):
        a=self.actor(torch.tensor(state).to(self.device))[0]
        b=[torch.distributions.Categorical(logits=x).sample().item() for x in a.T]
        act=np.zeros(a.shape,dtype='int32')
        act[b,range(act.shape[1])]=1
        assert (act.sum(axis=0)==1).all(),'act wrong'
        return act

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        n_p=self.actor(next_states)
        next_probs = FU.softmax(n_p,dim=1)
        next_log_probs = FU.log_softmax(n_p,dim=1)
        entropy = -torch.mean(torch.sum(next_probs * next_log_probs, dim=1, keepdim=False),dim=1,keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.mean(torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=False),dim=1,keepdim=True)
        next_value = (min_qvalue + self.log_alpha.exp() * entropy)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        self.writer.add_scalar('rewards_min',rewards.min().item(),self.step)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        F=lambda x:torch.tensor(x,dtype=torch.float32).to(self.device)
        states=F(np.concatenate(transition_dict['states'],axis=0))
        next_states=F(np.concatenate(transition_dict['next_states'],axis=0))
        actions=F(np.vstack(transition_dict['actions']).reshape(-1,*transition_dict['actions'][0].shape))
        assert (actions.sum(axis=1)).all(),'actions wrong'
        # print('actions',actions)
        rewards=F(transition_dict['rewards']).reshape(-1,1)
        overs=F(transition_dict['overs']).reshape(-1,1)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, overs)
        critic_1_q_values = (self.critic_1(states)*actions).sum(dim=1).mean(dim=1,keepdim=True)
        critic_1_loss = torch.mean(
            FU.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values =(self.critic_2(states)*actions).sum(dim=1).mean(dim=1,keepdim=True)
        critic_2_loss = torch.mean(
            FU.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # print(critic_1_q_values)
        # print(td_target)

        # 更新策略网络
        o_p=self.actor(states)
        probs = FU.softmax(o_p,dim=1)
        log_probs = FU.log_softmax(o_p,dim=1)
        # 直接根据概率计算熵
        entropy = -torch.mean(torch.sum(probs * log_probs, dim=1, keepdim=False),dim=1,keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.mean(torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=False),dim=1,keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        self.step+=1
        self.writer.add_scalar('min_qvalue',min_qvalue.mean().item(),self.step)
        self.writer.add_scalar('critic_1_loss',critic_1_loss,self.step)
        self.writer.add_scalar('critic_2_loss',critic_2_loss,self.step)
        self.writer.add_scalar('entropy',entropy.mean().item(),self.step)
        self.writer.add_scalar('actor_loss',actor_loss,self.step)
        self.writer.add_scalar('log_alpha',self.log_alpha,self.step)
        self.writer.add_scalar('alphg_grad',self.log_alpha.grad,self.step)
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in self.actor.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
        self.writer.add_scalar('a_grad_l2', grad_means / grad_count, self.step)
        self.writer.add_scalar('a_grad_max', grad_max, self.step)
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in self.critic_1.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
        self.writer.add_scalar('c_1_grad_l2', grad_means / grad_count, self.step)
        self.writer.add_scalar('c_1_grad_max', grad_max, self.step)
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in self.critic_2.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
        self.writer.add_scalar('c_2_grad_l2', grad_means / grad_count, self.step)
        self.writer.add_scalar('c_2_grad_max', grad_max, self.step)
        probs_new=self.actor(states)
        kl=((FU.log_softmax(probs_new,dim=1)-FU.log_softmax(probs,dim=1))*FU.softmax(probs_new,dim=1)).sum(dim=1).mean().item() # not accurate
        self.writer.add_scalar('KL', kl, self.step)
    
    def save_model(self,path):
        torch.save(self.actor,path)
    
    def load_model(self,path):
        self.actor=torch.load(path)