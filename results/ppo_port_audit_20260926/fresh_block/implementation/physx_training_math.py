"""Numerical helpers independent of the simulator and learner."""
import numpy as np

def continuous_joint_position(previous,current,accumulated):
    delta=np.asarray(current)-previous
    delta-=np.round(delta/(2*np.pi))*(2*np.pi)
    return accumulated+delta

def pd_effort(q,qd,target,kp,kd,limit):
    return np.clip(np.asarray(kp)*(target-q)-np.asarray(kd)*qd,-np.asarray(limit),np.asarray(limit))

def gae_returns(reward,value,done,last_value,*,gamma,lam):
    advantage=np.zeros_like(reward);carry=np.zeros_like(last_value);next_value=last_value
    for t in reversed(range(len(reward))):
        mask=1-done[t]
        carry=reward[t]+gamma*next_value*mask-value[t]+gamma*lam*mask*carry
        advantage[t]=carry;next_value=value[t]
    return advantage,advantage+value
