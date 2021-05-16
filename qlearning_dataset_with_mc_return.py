import numpy as np
from tqdm import tqdm
from collections import deque

GAMMA = 0.99

def compute_mc_returns(rewards):
    mc_return = 0
    mc_returns = deque()
    for r in reversed(rewards):
        mc_return = r + GAMMA * mc_return
        mc_returns.appendleft(mc_return)  # if list is used then O(n) operation
    return list(mc_returns)
    
def qlearning_dataset_with_mc_return(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    mc_returns_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    
    rewards_of_current_episode = []
    
    for i in tqdm(range(N)): 
        
        obs = dataset['observations'][i].astype(np.float32)
        if i + 1 > N - 1:  # N - 1 is the last timestep; so here we are asking, is i+1 an invalid index?
            new_obs = np.zeros_like(obs)
            # Reasoning on why this is the correct thing to do
            # At the very end, there are two possible scenarios
            # - no done flag: final_timestep=True, last transition is ignored (so this full of zeros next state is not used)
            # - yes done flag: since done=True, the value of the next state is not used
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        
        rewards_of_current_episode.append(reward)

        if use_timeouts:  # we are in this case, since all D4RL datasets have the "timeouts" field
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        
        if (not terminate_on_end) and final_timestep:  # if final_timestep, we are in this case
            
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            
            # last transition is not actually included in the dataset (no next state), but MC returns should consider it
            # so essentially [:-1] deal with the un-matched length of mc_returns and other stuff
            
            mc_returns = compute_mc_returns(rewards_of_current_episode)[:-1]
            #assert len(mc_returns) == 999
            mc_returns_.extend(mc_returns)
            rewards_of_current_episode = []
            
            continue  
        
        if done_bool or final_timestep:
            episode_step = 0
            mc_returns = compute_mc_returns(rewards_of_current_episode)
            mc_returns_.extend(mc_returns)
            rewards_of_current_episode = []
            
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'mc_returns': np.array(mc_returns_)
    }