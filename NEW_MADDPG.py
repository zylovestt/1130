import torch
import numpy as np
import torch.nn.functional as FU
import torch.nn.utils as nn_utils
from torch import nn
from torch.utils.tensorboard import SummaryWriter

FMakeNet_1=lambda x,y,z:[nn.PReLU() if not i%2 else nn.Linear(x,x) if i<y*2-1 else nn.Linear(x,z) for i in range(y*2)]

def onehot_from_logits(logits, eps):
    ''' 生成最优动作的独热(one-hot)形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作 
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return FU.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature,eps,hard):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y,eps)
        y = (y_hard.to(logits.device) - y).detach() + y
        # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
        # 正确地反传梯度
    return y

class MakeFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, net_deep=2):
        super().__init__()
        bn=[nn.Linear(num_in,hidden_dim)]
        bn.extend(FMakeNet_1(hidden_dim,net_deep,num_out))
        self.net=nn.Sequential(*bn)
        self.pros=num_out

    def __call__(self, x):
        mask=x[:,:self.pros]
        a=torch.zeros_like(mask)
        a[:]=1e8
        a*=(~(mask==1))
        return self.net(x)-a


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = MakeFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = MakeFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.actor_optimizer = torch.optim.NAdam(self.actor.parameters(),
                                                lr=actor_lr)
        

    def take_action(self, state, explore,eps,tem,log,hard):
        if log:
            action=self.actor(state)[0]
            act=torch.distributions.Categorical(logits=action).sample().item()
            return np.eye(action.shape[-1])[act]
        else:
            action = self.actor(state)
            if explore:
                action = gumbel_softmax(action,tem,eps,hard)
            else:
                action = onehot_from_logits(action,eps)
            return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau,act_epochs,
                 writer:SummaryWriter,clip_grad,log,hard):
        self.critic1 = MakeFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic1 = MakeFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.critic_optimizer1 = torch.optim.NAdam(self.critic1.parameters(),
                                                 lr=critic_lr)
        
        self.critic2 = MakeFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic2 = MakeFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer2 = torch.optim.NAdam(self.critic2.parameters(),
                                                 lr=critic_lr)
        
        self.agents = []
        for i in range(env.jf.tasknum):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.act_epochs=act_epochs
        self.writer=writer
        # self.eps=0.1
        self.eps=0.0
        # self.tem=2
        self.tem=1
        self.step=0
        self.clip_grad=clip_grad
        self.log=log
        self.hard=hard

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore=False):
        states = [
            torch.tensor(np.array([states[i]]), dtype=torch.float, device=self.device)
            for i in range(len(self.agents))]
        return [
            agent.take_action(state, explore,self.eps,self.tem,self.log,self.hard)
            for agent, state in zip(self.agents, states)]

    def update(self, sample, i_agent,num_update,cri):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        if cri: #  ((num_update%len(self.agents))==0):
            self.step+=1
            # if (self.step%50)==0:
            #     if self.eps>0.01:
            #         self.eps*=0.96
                # if self.tem>0.1: #0.5->0.1
                #     self.tem*=0.96
            c_epochs=3
            for i in range(c_epochs):
                self.critic_optimizer1.zero_grad()
                self.critic_optimizer2.zero_grad()
                if self.log:
                    all_target_act = [
                        torch.eye(act[0].shape[-1])[torch.distributions.Categorical(logits=(pi(_next_obs))).sample()]
                        for pi, _next_obs in zip(self.target_policies, next_obs)
                    ]
                else:
                    all_target_act = [
                        # gumbel_softmax(pi(_next_obs),self.tem,self.eps,self.hard) #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                        onehot_from_logits(pi(_next_obs),self.eps) #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                        for pi, _next_obs in zip(self.target_policies, next_obs)
                    ]
                target_critic_input = torch.cat((*next_obs[:1], *all_target_act), dim=1)
                target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * torch.min(
                    self.target_critic1(target_critic_input),self.target_critic2(target_critic_input)) #change
                critic_input = torch.cat((*obs[:1], *act), dim=1)
                critic_value1 = self.critic1(critic_input)
                critic_value2 = self.critic2(critic_input)
                critic_loss1 = self.critic_criterion(critic_value1,target_critic_value.detach())
                critic_loss1.backward()
                if not self.clip_grad=='max':
                    nn_utils.clip_grad_norm_(self.critic1.parameters(),self.clip_grad)
                self.critic_optimizer1.step()
                critic_loss2 = self.critic_criterion(critic_value2,target_critic_value.detach())
                critic_loss2.backward()
                if not self.clip_grad=='max':
                    nn_utils.clip_grad_norm_(self.critic2.parameters(),self.clip_grad)
                self.critic_optimizer2.step()
                self.writer.add_scalar('critic_loss1',critic_loss1,c_epochs*num_update+i)
                self.writer.add_scalar('critic_loss2',critic_loss2,c_epochs*num_update+i)

        if not cri: #      (((num_update//len(self.agents))%self.act_epochs)==0):
            cur_agent.actor_optimizer.zero_grad()
            cur_actor_out = cur_agent.actor(obs[i_agent])
            if self.log:
                all_act_now = [
                    torch.eye(act[0].shape[-1])[torch.distributions.Categorical(logits=(pi(_next_obs))).sample()]
                    for pi, _next_obs in zip(self.policies, next_obs)
                ]
                vf_in=torch.cat((*obs[:1],*all_act_now),dim=1)
                Q=self.target_critic1(vf_in)
                actor_loss1=-(((FU.log_softmax(cur_actor_out,dim=1)*all_act_now[i_agent]).sum(1,keepdim=True))*Q).mean()
                epo_loss=(FU.log_softmax(cur_actor_out,dim=1)*FU.softmax(cur_actor_out,dim=1)).sum(1,keepdim=True).mean()
                actor_loss=actor_loss1+epo_loss*1e-1
                self.writer.add_scalar('epo_loss'+str(i_agent),epo_loss,num_update)
            else:
                cur_act_vf_in = gumbel_softmax(cur_actor_out,self.tem,self.eps,self.hard)
                all_actor_acs = []
                for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
                    if i == i_agent:
                        all_actor_acs.append(cur_act_vf_in)
                    else:
                        # all_actor_acs.append(gumbel_softmax(pi(_obs),self.tem,self.eps,self.hard)) #@@@@@@@@@@@@@@@@@@@@@@
                        all_actor_acs.append(onehot_from_logits(pi(_obs),self.eps)) #@@@@@@@@@@@@@@@@@@@@@@
                vf_in = torch.cat((*obs[:1], *all_actor_acs), dim=1)
                # actor_loss = -torch.min(self.critic1(vf_in),self.critic2(vf_in)).mean()
                actor_loss = -self.critic1(vf_in).mean()
                # actor_loss = -((self.critic1(vf_in)+self.critic2(vf_in))/2).mean()
                actor_norm = (((cur_actor_out>-1e7).float()*cur_actor_out)**2).mean()
                # actor_norm = (cur_actor_out**2).mean()
                actor_loss += actor_norm * 1e-3
            actor_loss.backward()
            if not self.clip_grad=='max':
                nn_utils.clip_grad_norm_(cur_agent.actor.parameters(),self.clip_grad)
            cur_agent.actor_optimizer.step()
            self.writer.add_scalar('actor_loss'+str(i_agent),actor_loss,num_update)
            epo_loss=(FU.log_softmax(cur_actor_out,dim=1)*FU.softmax(cur_actor_out,dim=1)).sum(1,keepdim=True).mean()
            self.writer.add_scalar('epo_loss'+str(i_agent),epo_loss,num_update)
            self.writer.add_scalar('actor_norm'+str(i_agent),actor_norm,num_update)
    
    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(self.critic1, self.target_critic1, self.tau)
            agt.soft_update(self.critic2, self.target_critic2, self.tau)