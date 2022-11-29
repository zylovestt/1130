def fstate(f,state):
    if (type(state)==list or type(state)==tuple) and type(state[0])==dict:
        state={k:[x[k] for x in state] for k in state[0]}
    if type(state)==dict:
        return {k:f(state[k]) for k in state}
    return f(state)