import collections

import numpy as np
import gym
import d4rl

Trajectory = collections.namedtuple('Trajectory', 'states actions rewards dones frames')


def _parse_v0(env_id):
    env = gym.make(env_id)
    dataset = env.get_dataset()
    obs, acs, rs, dones =\
        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['terminals']

    def _parse(obs,actions,rewards,dones,trim_first_T,max_episode_steps):
        trajs = []
        start = trim_first_T
        while start < len(dones):
            end = start

            while end != 1000000 - 1 and end < len(dones) - 1 and \
                (not dones[end] and end - start + 1 < max_episode_steps):
                end += 1

            if dones[end]:
                # the trajectory ends normally.
                # since the next state will not be (should not be, actually) used by any algorithms,
                # we add null states (zero-states) at the end.

                traj = Trajectory(
                    states = np.concatenate([obs[start:end+1],np.zeros_like(obs[0])[None]],axis=0),
                    actions = actions[start:end+1],
                    rewards = rewards[start:end+1],
                    dones = dones[start:end+1].astype(np.bool_),
                    frames = None,
                )

                assert np.all(traj.dones[:-1] == False) and traj.dones[-1]

            else:
                # episodes end unintentionally (terminate due to timeout, cut-off when concateante two trajectories, or etc).
                # since the next-state is not available, it drops the last action.

                traj = Trajectory(
                    states = obs[start:end+1],
                    actions = actions[start:end],
                    rewards = rewards[start:end],
                    dones = dones[start:end].astype(np.bool_),
                    frames = None,
                )

                assert np.all(traj.dones == False)

            if len(traj.states) > 1: # some trajectories are extremely short in -medium-replay dataset (due to unexpected timeout caused by RLKIT); https://github.com/rail-berkeley/d4rl/issues/86#issuecomment-778566671
                trajs.append(traj)

            start = end + 1

        return trajs

    if env_id == 'halfcheetah-medium-replay-v0':
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)
    elif env_id == 'halfcheetah-medium-v0':
        trajs = _parse(obs,acs,rs,dones,899,env._max_episode_steps-1) # why env._max_episode_stpes - 1? it is questionable, but it looks a valid thing to do.
    elif env_id == 'halfcheetah-expert-v0':
        trajs = _parse(obs,acs,rs,dones,996,env._max_episode_steps-1)
    elif env_id == 'halfcheetah-medium-expert-v0':
        trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],899,env._max_episode_steps-1) + \
            _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],996,env._max_episode_steps-1)
    elif env_id == 'hopper-medium-v0':
        trajs = _parse(obs,acs,rs,dones,211,env._max_episode_steps)
    elif env_id == 'hopper-expert-v0':
        trajs = _parse(obs,acs,rs,dones,309,env._max_episode_steps-1)
    elif env_id == 'hopper-medium-expert-v0': # actually, expert + mixed
        trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],309,env._max_episode_steps-1) + \
            _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],0,env._max_episode_steps-1)
    elif env_id == 'walker2d-medium-v0':
        trajs = _parse(obs,acs,rs,dones,644,env._max_episode_steps)
    elif env_id == 'walker2d-expert-v0':
        trajs = _parse(obs,acs,rs,dones,487,env._max_episode_steps-1)
    elif env_id == 'walker2d-medium-expert-v0': # actually, expert + mixed
        trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],644,env._max_episode_steps) + \
            _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],487,env._max_episode_steps-1)
    elif env_id in ['halfcheetah-random-v0', 'walker2d-random-v0', 'hopper-random-v0', 'walker2d-medium-replay-v0', 'hopper-medium-replay-v0']:
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps-1)
    elif env_id in ['pen-expert-v0', 'hammer-expert-v0', 'door-expert-v0', 'relocate-expert-v0']:
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)
    elif env_id in ['door-human-v0','relocate-human-v0','hammer-human-v0']:
        trajs = _parse(obs,acs,rs,dones,0,1000)
        for traj in trajs:
            traj.dones[:] = False # this is philosophical decision; since its original env does not terminate, so 'done' in the human data does not meaning anything. I regard this information is given only as a trajectory separator.
    elif env_id in ['door-cloned-v0','relocate-cloned-v0','hammer-cloned-v0']:
        trajs = _parse(obs[:500000],acs[:500000],rs[:500000],dones[:500000],0,1000) + \
            _parse(obs[500000:],acs[500000:],rs[500000:],dones[500000:],0,env._max_episode_steps)
        for traj in trajs:
            traj.dones[:] = False # this is philosophical decision; since its original env does not terminate, so 'done' in the human data does not meaning anything. I regard this information is given only as a trajectory separator.
    elif env_id in ['pen-human-v0']:
        trajs = _parse(obs,acs,rs,np.zeros_like(dones),0,200)
        for traj in trajs:
            traj.dones[:] = False
    elif env_id in ['pen-cloned-v0']:
        trajs = _parse(obs[:250000],acs[:250000],rs[:250000],dones[:250000],0,200) + \
            _parse(obs[250000:],acs[250000:],rs[250000:],dones[250000:],0,env._max_episode_steps)
    else:
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)

    return trajs


def parse_S_A_R_D_NS_NA_from_trajs(trajs):
    
    s, a, r, d, ns, na = [], [], [], [], [], []
    
    action_dim = len(trajs[0].actions[0])
    
    for traj in trajs:
        
        traj_len = len(traj.rewards)
        
        for t in range(traj_len):
            
            if t == traj_len - 1:  # final timestep
                if traj.dones[t]:  # ok to append a dummy next action; done prevents bootstrapping
                    na.append(np.zeros((action_dim, )))
                else:              # the final timestep should be discarded
                    break          # start next trajectory
            else:
                na.append(traj.actions[t+1])
            
            s.append(traj.states[t])
            a.append(traj.actions[t])
            r.append(traj.rewards[t])
            d.append(traj.dones[t])
            ns.append(traj.states[t+1])
    
    return s, a, r, d, ns, na


def qlearning_dataset_with_next_action(env):
    trajs = _parse_v0(env)
    s, a, r, d, ns, na = parse_S_A_R_D_NS_NA_from_trajs(trajs)
    return {
        'observations': np.array(s),
        'actions': np.array(a),
        'rewards': np.array(r),
        'terminals': np.array(d),
        'next_observations': np.array(ns),
        'next_actions': np.array(na)
    }