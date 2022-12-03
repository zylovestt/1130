import numpy as np
import matplotlib.pyplot as plt

def model_test(seed,env,agent,num_episodes,plot=False):
    assert not (num_episodes>1 and plot)
    if plot:
        task_loc=[]
        pro_index=[]
    agent.rng=np.random.RandomState(seed)
    return_list = []
    env.set_test_mode(seed)
    for _ in range(num_episodes):
        episode_return = 0
        state = env.reset()
        # print(env.job.tasks,env.job.loc,env.job.time,env.job.real_tasknum)
        done = 0
        while not done:
            action = agent.take_action(state)
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
    return np.mean(return_list)