import numpy as np

def model_test(seed,env,agent,num_episodes):
    agent.rng=np.random.RandomState(seed)
    env.set_test_mode(seed)
    return_list = []
    for _ in range(num_episodes):
        episode_return = 0
        state = env.reset()
        # print(env.job.tasks,env.job.loc,env.job.time,env.job.real_tasknum)
        done = 0
        while not done:
            action = agent.take_action(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    env.set_train_mode()
    # print(np.mean(return_list))
    return np.mean(return_list)