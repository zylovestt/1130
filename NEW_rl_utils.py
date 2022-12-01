import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from NEW_TEST import model_test
from NEW_ENV2 import NEW_ENV
from copy import deepcopy
from NEW_TEST import model_test
import collections
import random
import torch.multiprocessing as mp
import time
import NEW_TD3

def mpp_model_test(seed,env,agent:NEW_TD3.TD3,num_episodes,queue:mp.Queue):
    agent.explore=False
    while True:
        model=queue.get()
        if model is None:
            break
        agent.actor.load_state_dict(model)
        agent.rng=np.random.RandomState(seed)
        env.set_test_mode(seed)
        return_list = []
        for _ in range(num_episodes):
            episode_return = 0
            state = env.reset()
            # print(env.job.tasks,env.job.loc,env.job.time,env.job.real_tasknum)
            done = 0
            while not done:
                with torch.no_grad():
                    action = agent.take_action(state)
                # print(action)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
        env.set_train_mode()
        print(np.mean(return_list))

def collect(env,agent,qin,qout):
    agent.explore=True
    done=None
    while True:
        s=qin.get()
        if s is None:
            break
        model,collect_num=s
        agent.actor.load_state_dict(model)
        steps=0
        l=[]
        t_c=time.time()
        while steps<collect_num:
            if done is None or done:
                state = env.reset()
                done = 0
            while not done and steps<collect_num:
                with torch.no_grad():
                    action = agent.take_action(state)
                next_state,reward,done,over = env.step(action)
                l.append((state, action, reward, next_state, done, over))
                state = next_state
                steps+=1
        qout.put(l)
        print('t_c',time.time()-t_c)
                    

def train_on_policy_agent(test_seed,env:NEW_ENV,agent,num_episodes,cal_steps,writer:SummaryWriter,test_cycles,test_epochs):
    return_list = []
    done_num=0
    gstep=0
    tstep=0
    subloop_num=int(num_episodes*env.maxsteps/(10*cal_steps))
    for i in range(10):
        with tqdm(total=subloop_num, desc='Iteration %d' % i) as pbar:
            episode_return = 0
            state = env.reset()
            done=0
            for i_episode in range(subloop_num):
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'overs': []}
                step=0
                while (not done) and (step<cal_steps):
                    step+=1
                    action = agent.take_action(state)
                    next_state, reward, done, over = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    assert (action.sum(axis=-1)==1).all(),'act wrong'
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    transition_dict['overs'].append(over)
                    state = next_state
                    episode_return += reward
                agent.update(transition_dict)
                if done:
                    done_num+=1
                    gstep+=1
                    writer.add_scalar('episode_return',episode_return,gstep)
                    return_list.append(episode_return)
                    episode_return = 0
                    done=0
                    if done_num % test_cycles == 0: #change
                        tstep+=1
                        agent.explore=False
                        test_return=model_test(test_seed,env,agent,test_epochs) #change
                        agent.explore=True
                        writer.add_scalar('test_return',test_return,tstep)
                        pbar.set_postfix({'episode': '%d' % (i * subloop_num + i_episode+1),
                                          'return': '%.3f' % np.mean(return_list[-10:]),
                                          'test_return': '%.3f' % test_return})
                    state = env.reset()
                pbar.update(1)
    return return_list

class ReplayBuffer:
    def __init__(self, capacity,device):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, over): 
        self.buffer.append((state, action, reward, next_state, done, over)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, over = zip(*transitions)
        # return np.array(state), action, reward, np.array(next_state), done, over
        return state, action, reward, next_state, done, over

    @property
    def size(self): 
        return len(self.buffer)

class Quick_ReplayBuffer:
    def __init__(self,capacity,device):
        self.capacity=capacity
        self.device=device
        self.buffer=None
    
    def add(self, state, action, reward, next_state, done, over):
        action=action.reshape(-1)
        if self.buffer is None:
            self.state_size=len(state)
            self.action_size=len(action)
            self.buffer=torch.empty((self.capacity,2*self.state_size+self.action_size+3),device=self.device)
            self.size=0
            self.p=0
        self.buffer[self.p]=torch.tensor(np.hstack([state,action,[reward],next_state,[done,over]]),device=self.device)
        if self.size<self.capacity:
            self.size+=1
        self.p=(self.p+1)%self.capacity
    
    def sample(self,batch_size):
        s=self.buffer[np.random.choice(range(self.size),batch_size)]
        sz=self.state_size
        sa=self.action_size
        state=s[:,:sz]
        action=s[:,sz:sz+sa]
        reward=s[:,sz+sa:sz+sa+1]
        next_state=s[:,sz+sa+1:2*sz+sa+1]
        done=s[:,-2:-1]
        over=s[:,-1:]
        return state, action, reward, next_state, done, over

class NEW_ReplayBuffer:
    def __init__(self,capacity,device):
        self.buffer = collections.deque(maxlen=capacity) 
        self.device=device
        self.flag=1
    
    def add(self, state, action, reward, next_state, done, over):
        action=action.reshape(-1)
        if self.flag:
            self.state_size=len(state)
            self.action_size=len(action)
            self.flag=0
        self.buffer.append(torch.tensor(np.hstack([state,action,[reward],next_state,[done,over]]),device=self.device).float())
    
    def sample(self,batch_size):
        s=torch.vstack(random.sample(self.buffer, batch_size))
        sz=self.state_size
        sa=self.action_size
        state=s[:,:sz]
        action=s[:,sz:sz+sa]
        reward=s[:,sz+sa:sz+sa+1]
        next_state=s[:,sz+sa+1:2*sz+sa+1]
        done=s[:,-2:-1]
        over=s[:,-1:]
        return state, action, reward, next_state, done, over
    
    @property
    def size(self): 
        return len(self.buffer)

def train_off_policy_agent(test_seed,env, agent, num_episodes, replay_buffer, minimal_size, batch_size,update_num,test_cycles,test_epochs):
    return_list = []
    a=0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, over = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done, over)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size >= minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_o = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'overs': b_o}
                        a+=1
                        if (a%update_num)==0:
                            agent.update(transition_dict)
                return_list.append(episode_return)
                if replay_buffer.size >= minimal_size and (i_episode+1) % test_cycles == 0:
                    agent.explore=False
                    with torch.no_grad():
                        test_return=model_test(test_seed,env,agent,test_epochs)
                    agent.explore=True
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'test_return': '%.3f' % test_return})
                pbar.update(1)
    return return_list

def ppf(x):
    print('hello world'+x)

def mp_train_off_policy_agent(test_seed,env, agent:NEW_TD3.TD3, num_episodes, replay_buffer, minimal_size, batch_size,update_num,test_cycles,test_epochs):
    data_proc_list=[]
    return_list = []
    a=0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, over = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done, over)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size >= minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_o = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'overs': b_o}
                        a+=1
                        if (a%update_num)==0:
                            agent.update(transition_dict)
                return_list.append(episode_return)
                if replay_buffer.size >= minimal_size and (i_episode+1) % test_cycles == 0:
                    agent.explore=False
                    # with torch.no_grad():
                    agent_mp=deepcopy(agent)
                    agent_mp.device='cpu'
                    agent_mp.actor.to('cpu')
                    agent_mp.critic1=agent_mp.critic2=agent_mp.target_critic1=agent_mp.target_critic2=agent_mp.target_actor=None
                    agent_mp.actor_optimizer=agent_mp.critic_optimizer1=agent_mp.critic_optimizer2=None
                    data_proc = mp.Process(target=model_test,args=(test_seed,env,agent_mp,test_epochs))
                    # x='zy'
                    # data_proc = mp.Process(target=ppf,args=(x,))
                    data_proc.start()
                    data_proc_list.append(data_proc)
                        # test_return=model_test(test_seed,env,agent,test_epochs)
                    agent.explore=True
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    for p in data_proc_list:
            p.terminate()
            p.join()
    return return_list

def mpp_train_off_policy_agent(test_seed,env, agent:NEW_TD3.TD3, num_episodes, replay_buffer, minimal_size, batch_size,update_num,test_cycles,test_epochs):
    agent.explore=True
    # agent.actor.share_memory()
    agent_mp=deepcopy(agent)
    agent_mp.device='cpu'
    agent_mp.actor.to('cpu')
    agent_mp.critic1=agent_mp.critic2=agent_mp.target_critic1=agent_mp.target_critic2=agent_mp.target_actor=None
    agent_mp.actor_optimizer=agent_mp.critic_optimizer1=agent_mp.critic_optimizer2=None
    queue=mp.Queue()
    data_proc = mp.Process(target=mpp_model_test,args=(test_seed,env,agent_mp,test_epochs,queue))
    data_proc.start()
    return_list = []
    a=0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, over = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done, over)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size >= minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_o = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'overs': b_o}
                        a+=1
                        if (a%update_num)==0:
                            agent.update(transition_dict)
                return_list.append(episode_return)
                if replay_buffer.size >= minimal_size and (i_episode+1) % test_cycles == 0:
                    
                    # with torch.no_grad():
                    
                    queue.put(deepcopy(agent.actor).to('cpu').state_dict())
                        # test_return=model_test(test_seed,env,agent,test_epochs)
                    
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    queue.put(None)
    return return_list

def mppp_train_off_policy_agent(test_seed,env, agent:NEW_TD3.TD3, num_episodes, replay_buffer, minimal_size, batch_size,update_num,test_cycles,test_epochs):
    agent.explore=True
    agent_mp=deepcopy(agent)
    agent_mp.device='cpu'
    agent_mp.actor.to('cpu')
    agent_mp.critic1=agent_mp.critic2=agent_mp.target_critic1=agent_mp.target_critic2=agent_mp.target_actor=None
    agent_mp.actor_optimizer=agent_mp.critic_optimizer1=agent_mp.critic_optimizer2=None
    queue=mp.Queue()
    data_proc = mp.Process(target=mpp_model_test,args=(test_seed,env,agent_mp,test_epochs,queue))
    data_proc.start()

    collect_queue_in=mp.Queue()
    collect_queue_out=mp.Queue()
    agent_collect=deepcopy(agent_mp)
    collect_proc=mp.Process(target=collect,args=(env,agent_collect,collect_queue_in,collect_queue_out))
    collect_proc.start()
    collect_queue_in.put((deepcopy(agent.actor).to('cpu').state_dict(),minimal_size))
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                collect_queue_in.put((deepcopy(agent.actor).to('cpu').state_dict(),update_num))
                collect_sample=collect_queue_out.get()
                for cs in collect_sample:
                    replay_buffer.add(*cs)
                assert replay_buffer.size >= minimal_size
                b_s, b_a, b_r, b_ns, b_d, b_o = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'overs': b_o}
                t_cuda=time.time()
                agent.update(transition_dict)
                print('cuda',time.time()-t_cuda)
                
                if (i_episode+1) % test_cycles == 0:
                    queue.put(deepcopy(agent.actor).to('cpu').state_dict())
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1)})
                pbar.update(1)
    queue.put(None)
    collect_queue_in.put(None)

def compute_advantage_batch(gamma, lmbda, td_delta,dones):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta,done in zip(td_delta[::-1],dones[::-1]):
        if done:
            advantage=0.0
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.concatenate(advantage_list,axis=0), dtype=torch.float)