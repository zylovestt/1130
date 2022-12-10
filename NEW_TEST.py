import numpy as np
import matplotlib.pyplot as plt
from PRINT import Logger
from NEW_ENV2 import NEW_ENV
from pprint import pprint

def model_test(seed,env:NEW_ENV,agent,num_episodes,plot=False,path='./Default.log'):
    assert not (num_episodes>1 and plot)
    logger=Logger(path)
    if plot:
        task_loc=[]
        pro_index=[]
    agent.rng=np.random.RandomState(seed)
    return_list = []
    env.set_test_mode(seed)
    for _ in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = 0
        while not done:
            if plot:
                pprint(env.pros.ps)
                pprint(env.job.tasks)
                print('delta_time',env.jf.delta_time)
            action = agent.take_action(state)
            if plot:
                print(action)
            if plot:
                task_loc.extend([(t['lx'],t['ly']) for t in env.job.tasks])
                pro_index.extend(np.argmax(action,axis=-1).tolist())
            # print(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    env.set_train_mode()
    if plot:
        task_loc=np.array(task_loc)
        pro_index=np.array(pro_index)
        fig=plt.figure()
        for i in range(len(action)):
            tl=task_loc[pro_index==i]
            if len(tl):
                plt.scatter(tl[:,0],tl[:,1],label=str(i))
        plt.legend()
        fig.savefig('trace')
    logger.reset()
    return np.mean(return_list)