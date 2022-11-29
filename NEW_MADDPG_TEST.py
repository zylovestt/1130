import torch
import numpy as np
from NEW_MADDPG import MADDPG
from NEW_PUBLIC_ENV import RandomAgent,model_test,env,device,writer
import collections
import random

class ReplayBuffer:
    def __init__(self, seed, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.rng=random.Random(seed)

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = self.rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self): 
        return len(self.buffer)

num_episodes = 10000
episode_length =  env.maxsteps # 每条序列的最大长度
buffer_size = 10000
hidden_dim = 500
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.95
tau = 1e-2
batch_size = 128
update_interval = 10
minimal_size = 1000
act_epochs=1
test_cycles=100
test_epochs=50


replay_buffer = ReplayBuffer(0, buffer_size)

state_dims = [env.state_size for _ in range(env.jf.tasknum)]
action_dims = [env.pf.num for _ in range(env.jf.tasknum)]  

critic_input_dim = env.state_size + sum(action_dims)

maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dims, critic_input_dim, gamma, tau,act_epochs,writer,1e-1,False,True)

def evaluate(test_seed,maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    # env = make_env(env_id)
    env.set_test_mode(test_seed)
    returns = np.zeros(env.jf.tasknum)
    for _ in range(n_episode):
        obs = env.reset(True)
        done=0
        while not done:
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions,True)
            done=done[0]
            rew = np.array(rew)
            returns += rew / n_episode
    env.set_train_mode()
    return returns.tolist()[0]


return_list = []  # 记录每一轮的回报（return）
total_step = 0
num_update = 0
# print(evaluate(0,maddpg, n_episode=4,episode_length=env.maxsteps))
for i_episode in range(num_episodes):
    state = env.reset(True)
    # ep_returns = np.zeros(len(env.agents))
    done=0
    while not done:
        actions = maddpg.take_action(state, explore=True)
        next_state, reward, done, _ = env.step(actions,True)
        replay_buffer.add(state, actions, reward, next_state, done)
        done=done[0]
        state = next_state
        total_step += 1
        if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
            sample = replay_buffer.sample(batch_size)
            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearranged]

            sample = [stack_array(x) for x in sample]
            num_update+=1
            maddpg.update(sample, 0, num_update,True)
            if not(num_update%2):
                for a_i in range(env.jf.tasknum):
                    if env.job.tasks_col['k'][a_i]:
                        maddpg.update(sample, a_i, num_update,False)
                # if True: #((num_update%env.jf.tasknum)==0) and  (((num_update//env.jf.tasknum)%act_epochs)==0):
                maddpg.update_all_targets()
    if replay_buffer.size() >= minimal_size and (i_episode + 1) % test_cycles == 0:
        ep_returns = evaluate(0,maddpg, n_episode=test_epochs)
        return_list.append(ep_returns)
        print(f"Episode: {i_episode+1}, {ep_returns}")

tseed=0
tnum=test_epochs
ra=RandomAgent(tseed,env.pros.num,env.jf.tasknum)
FTEST=lambda x:print(model_test(0,env,x,tnum))
FTEST(ra)
FTEST(ra)

print(evaluate(tseed,maddpg, n_episode=tnum,episode_length=env.maxsteps))
print(evaluate(tseed,maddpg, n_episode=tnum,episode_length=env.maxsteps))
FTEST(ra)